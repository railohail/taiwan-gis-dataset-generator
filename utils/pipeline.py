"""
Dataset generation pipeline.

This is the main processing module that orchestrates the complete dataset
generation workflow:

1. Separate district processing - each image contains ONE county/district
2. Combined maps processing - each image contains ALL districts
3. Augmentation pipeline - noise, hue shifts, rotation, windowing

The pipeline supports both COCO and YOLO output formats with optional
train/val/test splitting.
"""

import os
import cv2
import copy
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any, List

# Configuration and constants
from .config import (
    create_categories,
    COUNTY_TO_CLASS,
    COUNTY_ENGLISH_NAMES,
    normalize_county_name,
)

# Image processing utilities
from .image import (
    preprocess_raster,
    apply_distance_based_noise,
    apply_hue_augmentation,
    generate_hue_augmentation_params,
    convert_to_grayscale,
    unicode_safe_imwrite,
    rotate_image_and_annotations,
    scale_image_and_annotations,
    generate_random_angles,
)

# GIS/geometry utilities
from .geometry import (
    find_tif_files,
    load_data,
    extract_single_district_image,
    crop_image_and_shapefile,
    generate_windows,
    apply_mask_regions_to_shapefile,
)

# Annotation handling
from .annotations import (
    create_annotations_for_image,
    create_annotations_for_window,
    batch_append_to_coco_buffer,
    flush_coco_batch,
    flush_all_coco_batches,
    is_temp_file_mode_enabled,
    write_temp_annotation_file,
)

# Visualization
from .visualization import create_transparent_mask_visualization

# YOLO format writer
from .writers.yolo import batch_append_to_yolo_buffer, flush_yolo_batch


# Module-level counters with proper initialization and reset capability
class _ProcessingState:
    """Encapsulates mutable processing state to avoid function attribute hacks."""

    def __init__(self):
        self.image_counter = 1
        self.annotation_counter = 1

    def reset(self):
        """Reset counters to initial state. Call between processing runs."""
        self.image_counter = 1
        self.annotation_counter = 1


# Global processing state instance
_processing_state = _ProcessingState()


def reset_processing_state():
    """
    Reset all processing counters and state.

    Call this function before starting a new dataset generation run
    to ensure clean state and avoid ID conflicts.
    """
    _processing_state.reset()


# =============================================================================
# Helper Functions - Extract common patterns from processing functions
# =============================================================================

def _setup_output_directories(
    base_output_dir: str,
    use_split: bool
) -> Tuple[Optional[str], Optional[str]]:
    """
    Set up output directories based on configuration.

    Args:
        base_output_dir: Base output directory path
        use_split: Whether train/val/test split is enabled

    Returns:
        Tuple of (images_dir, visualizations_dir) - may be None if using splits
    """
    if not use_split:
        images_dir = os.path.join(base_output_dir, 'images')
        visualizations_dir = os.path.join(base_output_dir, 'visualizations')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        return images_dir, visualizations_dir
    else:
        # When using splits, directories will be created per-image
        return None, None


def _load_and_preprocess_raster(
    tif_file: str,
    shapefile_path: str,
    crop_factor: float = 0.0
) -> Optional[Tuple[Any, Any, Any, Any, Any]]:
    """
    Load raster data and preprocess it.

    Args:
        tif_file: Path to TIF file
        shapefile_path: Path to shapefile
        crop_factor: Crop factor to apply (0.0-1.0)

    Returns:
        Tuple of (raster, shapefile, image_data, transform, cropped_result) or None on failure
    """
    raster, shapefile_reprojected = load_data(tif_file, shapefile_path)
    if raster is None or shapefile_reprojected is None:
        return None

    # Preprocess raster
    raster_data = raster.read()
    image_data = preprocess_raster(raster_data)
    if image_data is None:
        raster.close()
        return None

    # Crop image and shapefile
    cropped_result = crop_image_and_shapefile(raster, image_data, shapefile_reprojected, crop_factor)
    if any(x is None for x in cropped_result):
        raster.close()
        return None

    cropped_image, cropped_transform, cropped_bounds, clipped_shapefile = cropped_result

    if clipped_shapefile.empty:
        raster.close()
        return None

    return raster, shapefile_reprojected, cropped_image, cropped_transform, clipped_shapefile


def _flush_batch_buffers(
    config: Dict[str, Any],
    base_output_dir: str,
    coco_file_path: str
) -> None:
    """
    Flush any remaining buffered data to disk.

    Args:
        config: Configuration dictionary
        base_output_dir: Base output directory path
        coco_file_path: Path to COCO annotations file
    """
    output_format = config.get('output', {}).get('format', 'coco').lower()
    annotation_type = config.get('output', {}).get('annotation_type', 'segmentation').lower()
    use_split = config.get('output', {}).get('use_split', False)

    if output_format == 'yolo':
        labels_dir = os.path.join(base_output_dir, 'labels')
        flush_yolo_batch(labels_dir, annotation_type, force=True)
    else:
        if use_split:
            flush_all_coco_batches(force=True)
        else:
            flush_coco_batch(coco_file_path, force=True)


def _update_processing_counts(
    images_created: int,
    annotations_created: int
) -> Tuple[int, int]:
    """
    Update global processing counters and return values for local tracking.

    Args:
        images_created: Number of images created in this batch
        annotations_created: Number of annotations created in this batch

    Returns:
        Tuple of (images_created, annotations_created) for convenience
    """
    _processing_state.image_counter += images_created
    _processing_state.annotation_counter += annotations_created
    return images_created, annotations_created


def _collect_tif_files(
    config: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    Collect all TIF files from configured districts.

    Args:
        config: Configuration dictionary

    Returns:
        List of (district, tif_file_path) tuples
    """
    mapdata_base_dir = config.get('mapdata_base_dir', 'datasets/MAPDATA')
    max_files = config['processing']['max_files_per_district']
    all_tif_files = []

    for district in config['districts']:
        tif_files = find_tif_files(district, mapdata_base_dir)
        if max_files and len(tif_files) > max_files:
            tif_files = tif_files[:max_files]
        all_tif_files.extend([(district, tif_file) for tif_file in tif_files])

    return all_tif_files


def _load_mask_regions(
    tif_file: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Load mask regions for a TIF file from the mask database.

    Args:
        tif_file: Path to TIF file

    Returns:
        List of mask region dicts or None if no masks
    """
    from .masks import get_mask_database

    mask_db = get_mask_database()
    tif_basename = os.path.basename(tif_file)
    return mask_db.get_masks(tif_basename)


def _adjust_mask_regions_for_crop(
    mask_regions: List[Dict[str, Any]],
    crop_x: int,
    crop_y: int
) -> List[Dict[str, Any]]:
    """
    Adjust mask regions for crop offset.

    Args:
        mask_regions: Original mask regions
        crop_x: X crop offset in pixels
        crop_y: Y crop offset in pixels

    Returns:
        Adjusted mask regions
    """
    adjusted = []
    for region in mask_regions:
        adjusted.append({
            'x': region['x'] - crop_x,
            'y': region['y'] - crop_y,
            'width': region['width'],
            'height': region['height']
        })
    return adjusted


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_separate_districts(
    config: Dict[str, Any],
    base_output_dir: str,
    coco_file_path: str
) -> Tuple[int, int]:
    """
    Process each district separately - each image contains only ONE district.

    Args:
        config: Configuration dictionary
        base_output_dir: Base output directory path
        coco_file_path: Path to COCO annotations file

    Returns:
        Tuple of (total_images_created, total_annotations_created)
    """
    print("\n=== PROCESSING SEPARATE DISTRICTS ===")
    print("Each image will contain ONLY ONE district, cropped from the full map")

    total_images = 0
    total_annotations = 0
    categories = create_categories()

    # Set up output directories using helper function
    use_split = config.get('output', {}).get('use_split', False)
    images_dir, visualizations_dir = _setup_output_directories(base_output_dir, use_split)

    # Process each district with progress bar
    district_pbar = tqdm(config['districts'], desc="Districts", unit="district", position=0)
    for district in district_pbar:
        district_pbar.set_description(f"District: {district}")

        # Find TIF files in district folder
        mapdata_base_dir = config.get('mapdata_base_dir', 'datasets/MAPDATA')
        tif_files = find_tif_files(district, mapdata_base_dir)

        if not tif_files:
            tqdm.write(f"  No TIF files found in {mapdata_base_dir}/{district}/")
            continue

        max_files = config['processing']['max_files_per_district']
        if max_files and len(tif_files) > max_files:
            tif_files = tif_files[:max_files]

        # Process each TIF file in the district with progress bar
        file_pbar = tqdm(tif_files, desc=f"  Files ({district})", unit="file", position=1, leave=False)
        for tif_file in file_pbar:
            try:
                file_pbar.set_description(f"  File: {os.path.basename(tif_file)[:30]}")

                # Load raster data
                raster, shapefile_reprojected = load_data(tif_file, config['shapefile_path'])
                if raster is None or shapefile_reprojected is None:
                    continue

                # Save the original raster transform early (before raster is closed)
                original_raster_transform = raster.transform

                # Preprocess raster
                raster_data = raster.read()
                image_data = preprocess_raster(raster_data)
                if image_data is None:
                    raster.close()
                    continue

                # Apply crop_factor to FULL image FIRST (before extracting separate districts)
                # This uses the same cropping logic as combined mode
                crop_factor = config.get('crop_factor', 0.0)
                from .geometry import crop_image_and_shapefile

                cropped_result = crop_image_and_shapefile(raster, image_data, shapefile_reprojected, crop_factor)
                if any(x is None for x in cropped_result):
                    raster.close()
                    continue

                cropped_full_image, cropped_full_transform, cropped_full_bounds, cropped_shapefile = cropped_result

                if crop_factor > 0:
                    tqdm.write(f"    Applied crop_factor={crop_factor:.2f} to full image: {raster.width}x{raster.height} -> {cropped_full_image.shape[1]}x{cropped_full_image.shape[0]}")

                # Use cropped shapefile for district extraction
                shapefile_reprojected = cropped_shapefile

                # Get unique counties in this raster (prioritize English names)
                possible_columns = ['COUNTYENG', 'COUNTYNAME', 'NAME', 'County', 'county', 'COUNTY', 'C_Name', '縣市名稱']
                counties_in_raster = set()

                # Use only the first column that exists (prioritizes English)
                for col in possible_columns:
                    if col in shapefile_reprojected.columns:
                        counties = shapefile_reprojected[col].dropna().astype(str).str.strip()
                        counties_in_raster.update(counties[counties != 'None'])
                        break  # Stop after first valid column to avoid duplicates

                tqdm.write(f"    Counties found: {list(counties_in_raster)}")

                # Process each county separately (SEPARATE DISTRICTS MODE)
                county_pbar = tqdm(list(counties_in_raster), desc="    Counties", unit="county", position=2, leave=False)
                for county_name_raw in county_pbar:
                    # Normalize county name (supports both Chinese and English input)
                    # normalize_county_name is imported at module level
                    county_name = normalize_county_name(county_name_raw)

                    if county_name in COUNTY_TO_CLASS:
                        county_pbar.set_description(f"    County: {county_name_raw[:20]}")

                        # Check if this county exists in the shapefile (use raw name for shapefile lookup)
                        county_check = shapefile_reprojected[shapefile_reprojected.apply(
                            lambda row: any(
                                str(row[col]).strip() == county_name_raw
                                for col in ['COUNTYNAME', 'NAME', 'County', 'county', '縣市名稱', 'COUNTY', 'COUNTYENG', 'C_Name']
                                if col in shapefile_reprojected.columns and row[col] is not None
                            ), axis=1
                        )]

                        if county_check.empty:
                            tqdm.write(f"      WARNING: County {county_name_raw} not found in shapefile")
                            continue

                        # Extract district-specific image (from already cropped full image)
                        # Use raw name since extract function looks up in shapefile
                        # NO crop_factor here - already applied to full image above
                        # Use minimal buffer to reduce gray border in visualizations
                        district_data = extract_single_district_image(
                            raster, shapefile_reprojected, county_name_raw,
                            buffer_pixels=10, crop_factor=0.0  # Minimal buffer to avoid border artifacts
                        )

                        if district_data[0] is None:
                            continue

                        district_raster_data, district_transform, district_bounds, district_shapefile = district_data
                        district_image = preprocess_raster(district_raster_data)

                        if district_image is None:
                            continue

                        # Load mask regions from database (SEPARATE MODE)
                        from .masks import get_mask_database
                        from .geometry import apply_mask_regions_to_shapefile

                        mask_db = get_mask_database()
                        tif_basename = os.path.basename(tif_file)
                        mask_regions = mask_db.get_masks(tif_basename)

                        # Apply masks to shapefile geometry (creates holes in polygons)
                        if mask_regions:
                            tqdm.write(f"      Found {len(mask_regions)} mask region(s) for {tif_basename}")

                            # Calculate mask coverage and check threshold BEFORE applying
                            from shapely.geometry import box
                            from shapely.ops import unary_union

                            # Get district image bounds in geographic coordinates
                            height_px, width_px = district_image.shape[:2]
                            district_image_bounds_geo = box(*district_bounds)

                            # Convert mask regions to geographic polygons (using original transform)
                            mask_polygons_geo = []
                            for region in mask_regions:
                                x1_px, y1_px = region['x'], region['y']
                                x2_px, y2_px = x1_px + region['width'], y1_px + region['height']
                                x1_geo, y1_geo = raster.transform * (x1_px, y1_px)
                                x2_geo, y2_geo = raster.transform * (x2_px, y2_px)
                                mask_poly = box(
                                    min(x1_geo, x2_geo), min(y1_geo, y2_geo),
                                    max(x1_geo, x2_geo), max(y1_geo, y2_geo)
                                )
                                mask_polygons_geo.append(mask_poly)

                            combined_mask = unary_union(mask_polygons_geo)

                            # Calculate intersection with district image bounds
                            district_area = district_image_bounds_geo.area
                            mask_intersection_area = district_image_bounds_geo.intersection(combined_mask).area
                            coverage_pct = (mask_intersection_area / district_area * 100.0) if district_area > 0 else 0.0

                            tqdm.write(f"      Mask coverage in this district: {coverage_pct:.1f}%")

                            # Check skip threshold
                            mask_skip_threshold = config.get('processing', {}).get('mask_skip_threshold_separate', 50.0)
                            if coverage_pct > mask_skip_threshold:
                                tqdm.write(f"      SKIPPED {county_name_raw} - {coverage_pct:.1f}% masked (threshold: {mask_skip_threshold}%)")
                                continue

                            # Apply masks to shapefile (subtracts mask regions from polygons)
                            # Masks are in original image coords, need to use original raster.transform
                            district_shapefile = apply_mask_regions_to_shapefile(
                                district_shapefile,
                                mask_regions,
                                raster.transform
                            )

                            if district_shapefile is None or district_shapefile.empty:
                                tqdm.write(f"      SKIPPED {county_name_raw} - all areas masked")
                                continue

                            mask_regions = None  # Clear so they're not applied again to annotations

                        # Clean county name for filename - use English names
                        english_name = COUNTY_ENGLISH_NAMES.get(county_name, county_name)
                        clean_county_name = english_name.replace(' ', '_').replace('County', 'Co').replace('City', 'Ci')
                        file_base_name = f"separate_{Path(tif_file).stem}_{clean_county_name}"

                        # Process with different augmentations
                        images_created, annotations_created = process_single_district_image(
                            district_image, district_shapefile, district_transform,
                            config, file_base_name, images_dir, visualizations_dir,
                            coco_file_path, categories,
                            _processing_state.image_counter,
                            _processing_state.annotation_counter,
                            mode="separate",
                            base_output_dir=base_output_dir,
                            source_file_path=tif_file,
                            mask_regions=mask_regions,
                            raster_transform=original_raster_transform,
                        )

                        total_images += images_created
                        total_annotations += annotations_created
                        _processing_state.image_counter += images_created
                        _processing_state.annotation_counter += annotations_created

                # Close raster
                raster.close()

            except Exception as e:
                print(f"    Error processing {tif_file}: {e}")
                # Clean up on error
                if 'raster' in locals():
                    try:
                        raster.close()
                    except (OSError, AttributeError):
                        pass  # Raster may already be closed or invalid
                continue

    # Flush any remaining buffered data using helper function
    _flush_batch_buffers(config, base_output_dir, coco_file_path)

    print(f"\nSeparate districts processing complete: {total_images} images, {total_annotations} annotations")
    return total_images, total_annotations


def process_combined_maps(
    config: Dict[str, Any],
    base_output_dir: str,
    coco_file_path: str
) -> Tuple[int, int]:
    """
    Process all districts combined in single images.

    Args:
        config: Configuration dictionary
        base_output_dir: Base output directory path
        coco_file_path: Path to COCO annotations file

    Returns:
        Tuple of (total_images_created, total_annotations_created)
    """
    print("\n=== PROCESSING COMBINED MAPS ===")
    print("Each image will contain ALL districts together (like full_coco_v1)")

    # Use shared processing state (counters continue from separate_districts if run)
    total_images = 0
    total_annotations = 0
    categories = create_categories()

    # Set up output directories using helper function
    use_split = config.get('output', {}).get('use_split', False)
    images_dir, visualizations_dir = _setup_output_directories(base_output_dir, use_split)

    # Collect all TIF files from all districts using helper function
    all_tif_files = _collect_tif_files(config)
    print(f"Processing {len(all_tif_files)} files from {len(config['districts'])} districts")

    # Process each TIF file (keeping all districts combined) with progress bar
    combined_pbar = tqdm(all_tif_files, desc="Combined Maps", unit="file", position=0)
    for district, tif_file in combined_pbar:
        try:
            combined_pbar.set_description(f"Combined: {os.path.basename(tif_file)[:40]}")

            # Load raster data
            raster, shapefile_reprojected = load_data(tif_file, config['shapefile_path'])
            if raster is None or shapefile_reprojected is None:
                continue

            # Save the original raster transform early (before raster is closed)
            original_raster_transform = raster.transform

            # Preprocess raster
            raster_data = raster.read()
            image_data = preprocess_raster(raster_data)
            if image_data is None:
                raster.close()
                continue

            # Crop image and shapefile (KEEP ALL DISTRICTS - like v1)
            cropped_result = crop_image_and_shapefile(raster, image_data, shapefile_reprojected, config['crop_factor'])
            if any(x is None for x in cropped_result):
                raster.close()
                continue

            cropped_image, cropped_transform, cropped_bounds, clipped_shapefile = cropped_result

            if clipped_shapefile.empty:
                raster.close()
                continue

            # Load mask regions from database using helper function
            mask_regions = _load_mask_regions(tif_file)

            # Apply masks to shapefile geometry (creates holes in polygons)
            if mask_regions:
                tif_basename = os.path.basename(tif_file)
                tqdm.write(f"    Found {len(mask_regions)} mask region(s) for {tif_basename}")

                # Adjust mask regions for crop offset using helper function
                crop_factor = config.get('crop_factor', 0.05)
                crop_x = int(raster.width * crop_factor)
                crop_y = int(raster.height * crop_factor)
                adjusted_mask_regions = _adjust_mask_regions_for_crop(mask_regions, crop_x, crop_y)

                # Apply masks to shapefile (subtracts mask regions from polygons)
                clipped_shapefile = apply_mask_regions_to_shapefile(
                    clipped_shapefile,
                    adjusted_mask_regions,
                    cropped_transform
                )

                if clipped_shapefile is None or clipped_shapefile.empty:
                    tqdm.write(f"    Skipping {tif_basename} - all areas masked")
                    raster.close()
                    continue

                mask_regions = None  # Clear mask_regions so they're not applied again to annotations

            file_base_name = f"combined_{Path(tif_file).stem}"

            # Process with different augmentations (keeping all districts)
            images_created, annotations_created = process_single_district_image(
                cropped_image, clipped_shapefile, cropped_transform,
                config, file_base_name, images_dir, visualizations_dir,
                coco_file_path, categories,
                _processing_state.image_counter,
                _processing_state.annotation_counter,
                mode="combined",
                base_output_dir=base_output_dir,
                source_file_path=tif_file,
                mask_regions=mask_regions,
                raster_transform=original_raster_transform,
            )

            total_images += images_created
            total_annotations += annotations_created
            _processing_state.image_counter += images_created
            _processing_state.annotation_counter += annotations_created

            raster.close()

        except Exception as e:
            print(f"    Error processing combined {tif_file}: {e}")
            continue

    # Flush any remaining buffered data using helper function
    _flush_batch_buffers(config, base_output_dir, coco_file_path)

    print(f"\nCombined maps processing complete: {total_images} images, {total_annotations} annotations")
    return total_images, total_annotations


def process_single_district_image(image, shapefile, transform, config, file_base_name,
                                images_dir, visualizations_dir, coco_file_path, categories,
                                start_image_id, start_annotation_id, mode="separate", base_output_dir=None,
                                source_file_path=None, mask_regions=None, raster_transform=None,
                                original_shapefile=None):
    """
    Process a single district image with all augmentations.

    Args:
        source_file_path: Path to the source TIF file (used for deterministic split assignment)
        mask_regions: List of mask region dicts (optional) for filtering annotations
        raster_transform: Original raster transform for mask coordinate conversion (required if mask_regions provided)
        original_shapefile: Copy of shapefile BEFORE mask subtraction (for visualization)
    """
    images_created = 0
    annotations_created = 0
    current_image_id = start_image_id
    current_annotation_id = start_annotation_id
    mask_triptych_created = False

    # Get output format and grayscale settings
    output_format = config.get('output', {}).get('format', 'coco').lower()
    annotation_type = config.get('output', {}).get('annotation_type', 'segmentation').lower()
    grayscale_enabled = config.get('output', {}).get('grayscale', False)
    use_split = config.get('output', {}).get('use_split', True)  # Works for both YOLO and COCO now

    # Determine split for this image (train/val/test)
    if use_split:
        from .writers.yolo import get_split_for_file, get_split_directories
        # Use passed base_output_dir or fall back to deriving it from images_dir
        if base_output_dir is None:
            base_output_dir = os.path.dirname(os.path.dirname(images_dir))  # Go up from train/images to root

        # Use deterministic split based on source file path
        if source_file_path:
            split_name = get_split_for_file(source_file_path)
        else:
            # Fallback: default to train with warning
            print(f"  WARNING: No source_file_path provided for split assignment. Defaulting to train.")
            split_name = 'train'

        split_images_dir, split_labels_dir = get_split_directories(base_output_dir, split_name, format_type=output_format)

        # Set visualization directory per split
        # For COCO format (Coconuts-1 style), create visualizations subdirectory in split folder
        if output_format == 'coco':
            split_visualizations_dir = os.path.join(split_images_dir, 'visualizations')
        else:
            split_visualizations_dir = os.path.join(base_output_dir, split_name, 'visualizations')
        os.makedirs(split_visualizations_dir, exist_ok=True)
    else:
        split_images_dir = images_dir
        split_labels_dir = os.path.join(os.path.dirname(images_dir), 'labels') if output_format == 'yolo' else None
        split_name = None
        split_visualizations_dir = visualizations_dir  # Use the passed-in visualizations_dir

    labels_dir = split_labels_dir
    # Override visualizations_dir to use split-specific directory
    visualizations_dir = split_visualizations_dir

    # Determine COCO file path for this split (if applicable)
    if use_split and split_name and output_format == 'coco':
        # Coconuts-1 style: _annotations.coco.json directly in split folder
        # Normalize split name for COCO format
        split_folder_name = 'valid' if split_name == 'val' else split_name
        active_coco_file = os.path.join(split_images_dir, '_annotations.coco.json')
        # Also set temp dir for fast COCO mode
        active_temp_dir = os.path.join(split_images_dir, 'temp_annotations')
    else:
        active_coco_file = coco_file_path
        # Set temp dir for non-split mode
        if base_output_dir:
            active_temp_dir = os.path.join(base_output_dir, 'temp_annotations')
        else:
            active_temp_dir = os.path.join(os.path.dirname(images_dir), 'temp_annotations')

    # Scale image if needed
    scaled_image, _, scaled_transform, scale_factor = scale_image_and_annotations(
        image, [], transform, max_size=1024
    )

    # Process each noise configuration with progress bar
    # Filter noise configs based on split: validation and test should only use clean noise
    all_noise_configs = [nc for nc in config['noise_configs'] if nc.get('enabled', True)]

    if use_split and split_name in ['val', 'test']:
        # For validation and test splits, only use clean noise (no augmentation)
        clean_noise = [nc for nc in all_noise_configs if nc['name'] == 'clean']
        if clean_noise:
            noise_configs = clean_noise
        else:
            # If no clean noise config exists, create a default one
            noise_configs = [{
                'name': 'clean',
                'intensity': 0.0,
                'type': 'gaussian',
                'acceleration': 0,
                'border_buffer_pixels': 0,
                'enabled': True
            }]
    else:
        # For training split (or no split), use all noise configurations
        noise_configs = all_noise_configs

    noise_pbar = tqdm(noise_configs, desc=f"      Noise ({mode})", unit="noise", position=3, leave=False)
    for noise_config in noise_pbar:
        noise_pbar.set_description(f"      Noise: {noise_config['name']}")

        # Apply noise
        if noise_config['name'] != 'clean':
            noisy_image, _ = apply_distance_based_noise(
                scaled_image, shapefile, scaled_transform, noise_config
            )
        else:
            noisy_image = scaled_image

        # Create base annotations from FULL shapefile (no pre-cutting)
        base_annotations = create_annotations_for_image(
            shapefile, noisy_image.shape, scaled_transform, current_image_id
        )

        # Apply mask filtering if masks exist (NEW APPROACH)
        # Subtracts mask rectangles from annotation polygons, creating holes
        if mask_regions and raster_transform:
            from .masks import filter_annotations_with_masks
            base_annotations, stats = filter_annotations_with_masks(
                base_annotations,
                mask_regions,
                scaled_transform,
                raster_transform,
                noisy_image.shape[1],  # width
                noisy_image.shape[0]   # height
            )
            # Only print stats if annotations were actually filtered
            if stats['filtered'] > 0 or stats['clipped'] > 0:
                print(f"      Mask filtering: {stats['kept']} kept, {stats['clipped']} clipped, {stats['filtered']} removed")

        # Update annotation IDs
        for ann in base_annotations:
            ann['id'] = current_annotation_id
            current_annotation_id += 1

        # Apply grayscale if enabled
        final_image = convert_to_grayscale(noisy_image) if grayscale_enabled else noisy_image

        # Save original image
        image_name = f'{file_base_name}_noise_{noise_config["name"]}_original.png'
        image_path = os.path.join(split_images_dir, image_name)

        # Save image using cv2 to maintain exact dimensions (no verification for speed)
        success, actual_width, actual_height = unicode_safe_imwrite(
            image_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR), verify=False
        )

        if not success:
            print(f"    Warning: Failed to save {image_name}")
            actual_height, actual_width = noisy_image.shape[:2]

        # Create image info with verified dimensions
        image_info = {
            "id": current_image_id,
            "width": actual_width,
            "height": actual_height,
            "file_name": image_name
        }

        # Create visualization (skip debug for performance)
        if config['visualization']['create_masks']:
            mask_name = f'{file_base_name}_noise_{noise_config["name"]}_original_masks.png'
            mask_path = os.path.join(visualizations_dir, mask_name)
            title = f'{mode.title()} - {noise_config["name"]} noise - {file_base_name}'
            # Use annotation-based visualization (same as legacy GIS_GEN)
            from .visualization import create_transparent_mask_visualization
            create_transparent_mask_visualization(noisy_image, base_annotations, categories, mask_path, title, create_debug=False)

            if (mask_regions and original_shapefile is not None and raster_transform is not None
                    and not mask_triptych_created):
                triptych_name = f'{file_base_name}_noise_{noise_config["name"]}_mask_triptych.png'
                triptych_path = os.path.join(visualizations_dir, triptych_name)
                create_mask_triptych_visualization(
                    noisy_image,
                    original_shapefile,
                    shapefile,
                    mask_regions,
                    scaled_transform,
                    triptych_path,
                    raster_transform=raster_transform,
                    title_prefix=title
                )
                mask_triptych_created = True

        # Add to batch buffer instead of writing immediately
        from .annotations import batch_append_to_coco_buffer, flush_coco_batch, is_temp_file_mode_enabled, write_temp_annotation_file
        try:
            # Handle annotations based on output format
            if output_format == 'yolo' and labels_dir:
                # Write YOLO format
                batch_append_to_yolo_buffer(image_info, base_annotations)
                flush_yolo_batch(labels_dir, annotation_type, force=False)
            else:
                # COCO format - use temp file mode if enabled (FAST)
                if is_temp_file_mode_enabled():
                    # Fast path: write individual temp JSON file per image
                    write_temp_annotation_file(image_info, base_annotations, active_temp_dir, annotation_type)
                else:
                    # Legacy path: use batch buffer with periodic JSON rewrites
                    batch_append_to_coco_buffer([image_info], base_annotations, annotation_type, coco_file_path=active_coco_file)
                    flush_coco_batch(active_coco_file, force=False)

            images_created += 1
            annotations_created += len(base_annotations)
        except Exception as e:
            print(f"      Error buffering annotations: {e}")
            import traceback
            traceback.print_exc()

        current_image_id += 1

        # Apply hue augmentation if enabled
        if config.get('hue_augmentation', {}).get('enabled', False):
            hue_config = config['hue_augmentation']
            hue_params = generate_hue_augmentation_params(
                hue_config['count'],
                hue_config['hue_shift_range'],
                hue_config['saturation_range'],
                hue_config['value_range']
            )

            for hue_idx, (hue_shift, sat_factor, val_factor) in enumerate(hue_params):
                # Apply hue augmentation
                hue_image = apply_hue_augmentation(noisy_image, hue_shift, sat_factor, val_factor)

                # Apply grayscale if enabled
                hue_final_image = convert_to_grayscale(hue_image) if grayscale_enabled else hue_image

                # Create annotations for hue-augmented image
                hue_annotations = create_annotations_for_image(
                    shapefile, hue_image.shape, scaled_transform, current_image_id
                )

                # Apply mask filtering if masks exist (NEW APPROACH - HUE)
                # Subtracts mask rectangles from annotation polygons, creating holes
                if mask_regions and raster_transform:
                    from .masks import filter_annotations_with_masks
                    hue_annotations, stats = filter_annotations_with_masks(
                        hue_annotations,
                        mask_regions,
                        scaled_transform,
                        raster_transform,
                        hue_image.shape[1],  # width
                        hue_image.shape[0]   # height
                    )
                    # Only print stats if annotations were actually filtered
                    if stats['filtered'] > 0 or stats['clipped'] > 0:
                        print(f"      Hue {hue_idx+1} mask filtering: {stats['kept']} kept, {stats['clipped']} clipped, {stats['filtered']} removed")

                # Update annotation IDs
                for ann in hue_annotations:
                    ann['id'] = current_annotation_id
                    current_annotation_id += 1

                # Save hue-augmented image (no verification for speed)
                hue_image_name = f'{file_base_name}_noise_{noise_config["name"]}_hue_{hue_idx+1}.png'
                hue_image_path = os.path.join(split_images_dir, hue_image_name)

                success, hue_actual_width, hue_actual_height = unicode_safe_imwrite(
                    hue_image_path, cv2.cvtColor(hue_final_image, cv2.COLOR_RGB2BGR), verify=False
                )

                if not success:
                    hue_actual_height, hue_actual_width = hue_image.shape[:2]

                # Create hue image info with verified dimensions
                hue_image_info = {
                    "id": current_image_id,
                    "width": hue_actual_width,
                    "height": hue_actual_height,
                    "file_name": hue_image_name
                }

                # Create hue visualization (skip debug for performance)
                if config['visualization']['create_masks']:
                    hue_mask_name = f'{file_base_name}_noise_{noise_config["name"]}_hue_{hue_idx+1}_masks.png'
                    hue_mask_path = os.path.join(visualizations_dir, hue_mask_name)
                    hue_title = f'{mode.title()} Hue {hue_idx+1} - {noise_config["name"]} - {file_base_name}'
                    create_transparent_mask_visualization(hue_image, hue_annotations, categories, hue_mask_path, hue_title, create_debug=False)

                # Add to batch buffer
                try:
                    if output_format == 'yolo' and labels_dir:
                        batch_append_to_yolo_buffer(hue_image_info, hue_annotations)
                        flush_yolo_batch(labels_dir, annotation_type, force=False)
                    else:
                        batch_append_to_coco_buffer([hue_image_info], hue_annotations, annotation_type, coco_file_path=active_coco_file)
                        flush_coco_batch(active_coco_file, force=False)
                    images_created += 1
                    annotations_created += len(hue_annotations)
                except Exception as e:
                    print(f"        Error buffering hue image annotations: {e}")

                current_image_id += 1

        # Apply rotation if enabled
        if config.get('rotation', {}).get('enabled', False):
            rotation_config = config['rotation']
            rotation_angles = generate_random_angles(rotation_config['count'], rotation_config['angle_range'])

            for rot_idx, angle in enumerate(rotation_angles):
                # Rotate image and annotations - SIMPLE approach without padding/clipping
                # Just rotate and keep the new dimensions
                annotations_copy = copy.deepcopy(base_annotations)
                rotated_image, rotated_annotations, (new_width, new_height) = rotate_image_and_annotations(
                    noisy_image, annotations_copy, angle,
                    rotation_config['interpolation'],
                    rotation_config['fill_value'],
                    defer_clipping=False  # Clip immediately to rotated image bounds
                )

                # That's it! No padding, no complex coordinate adjustments
                final_width, final_height = new_width, new_height

                # Apply grayscale if enabled
                rot_final_image = convert_to_grayscale(rotated_image) if grayscale_enabled else rotated_image

                # Update annotation IDs
                for ann in rotated_annotations:
                    ann['id'] = current_annotation_id
                    ann['image_id'] = current_image_id
                    current_annotation_id += 1

                # Save rotated image (no verification for speed)
                rot_image_name = f'{file_base_name}_noise_{noise_config["name"]}_rot_{rot_idx+1}.png'
                rot_image_path = os.path.join(split_images_dir, rot_image_name)

                success, rot_actual_width, rot_actual_height = unicode_safe_imwrite(
                    rot_image_path, cv2.cvtColor(rot_final_image, cv2.COLOR_RGB2BGR), verify=False
                )

                if not success:
                    rot_actual_height, rot_actual_width = rotated_image.shape[:2]

                # Create rotated image info with verified dimensions
                rotated_image_info = {
                    "id": current_image_id,
                    "width": rot_actual_width,
                    "height": rot_actual_height,
                    "file_name": rot_image_name
                }

                # Create rotated visualization (skip debug for performance)
                if config['visualization']['create_masks']:
                    rot_mask_name = f'{file_base_name}_noise_{noise_config["name"]}_rot_{rot_idx+1}_masks.png'
                    rot_mask_path = os.path.join(visualizations_dir, rot_mask_name)
                    rot_title = f'{mode.title()} Rot {rot_idx+1} ({angle:.1f}°) - {noise_config["name"]} - {file_base_name}'
                    create_transparent_mask_visualization(rotated_image, rotated_annotations, categories, rot_mask_path, rot_title, create_debug=False)

                # Add to batch buffer
                try:
                    if output_format == 'yolo' and labels_dir:
                        batch_append_to_yolo_buffer(rotated_image_info, rotated_annotations)
                        flush_yolo_batch(labels_dir, annotation_type, force=False)
                    else:
                        batch_append_to_coco_buffer([rotated_image_info], rotated_annotations, annotation_type, coco_file_path=active_coco_file)
                        flush_coco_batch(active_coco_file, force=False)
                    images_created += 1
                    annotations_created += len(rotated_annotations)
                except Exception as e:
                    print(f"        Error buffering rotated image annotations: {e}")

                current_image_id += 1

        # Apply windowing only for combined mode (like v1)
        if mode == "combined":
            print(f"    Processing windows for {mode} mode with {len(config['window_configs'])} window configurations")

            # Process each window configuration using the scaled full image
            for window_idx, window_config in enumerate(config['window_configs']):
                print(f"      Processing window config: {window_config['name']}")

                # Generate windows from scaled full image
                windows, window_info = generate_windows(noisy_image, window_config)

                # Process each window
                for win_idx, (window, w_info) in enumerate(zip(windows, window_info)):

                    # Create base annotations for this window using scaled transform
                    base_annotations = create_annotations_for_window(shapefile, w_info, scaled_transform)

                    # Apply mask filtering if masks exist (NEW APPROACH - WINDOWS)
                    # Subtracts mask rectangles from annotation polygons, creating holes
                    if mask_regions and raster_transform:
                        from .masks import filter_annotations_with_masks
                        # Use window transform (offset from scaled_transform by window position)
                        window_transform = rasterio.Affine(
                            scaled_transform.a, scaled_transform.b,
                            scaled_transform.c + scaled_transform.a * w_info['start_x'],
                            scaled_transform.d, scaled_transform.e,
                            scaled_transform.f + scaled_transform.e * w_info['start_y']
                        )
                        base_annotations, stats = filter_annotations_with_masks(
                            base_annotations,
                            mask_regions,
                            window_transform,
                            raster_transform,
                            window.shape[1],  # width
                            window.shape[0]   # height
                        )
                        # Only print stats if annotations were actually filtered
                        if stats['filtered'] > 0 or stats['clipped'] > 0:
                            print(f"        Window {win_idx+1} mask filtering: {stats['kept']} kept, {stats['clipped']} clipped, {stats['filtered']} removed")

                    # Update annotation IDs and image IDs
                    for ann in base_annotations:
                        ann['id'] = current_annotation_id
                        ann['image_id'] = current_image_id
                        current_annotation_id += 1

                    # Apply grayscale if enabled
                    window_final_image = convert_to_grayscale(window) if grayscale_enabled else window

                    # Save original window image (no verification for speed)
                    config_name = f"noise_{noise_config['name']}_window_{window_config['name']}"
                    window_image_name = f'{file_base_name}_win_{win_idx+1}_{config_name}_original.png'
                    window_image_path = os.path.join(split_images_dir, window_image_name)

                    success, window_actual_width, window_actual_height = unicode_safe_imwrite(
                        window_image_path, cv2.cvtColor(window_final_image, cv2.COLOR_RGB2BGR), verify=False
                    )

                    if not success:
                        window_actual_height, window_actual_width = window.shape[:2]

                    # Create window image info with verified dimensions
                    image_info = {
                        "id": current_image_id,
                        "width": window_actual_width,
                        "height": window_actual_height,
                        "file_name": window_image_name
                    }

                    # Create visualization (skip debug for performance)
                    if config['visualization']['create_masks']:
                        mask_name = f'{file_base_name}_win_{win_idx+1}_{config_name}_original_masks.png'
                        mask_path = os.path.join(visualizations_dir, mask_name)
                        title = f'{mode.title()} Window {win_idx+1} - {noise_config["name"]} noise, {window_config["name"]} window - {file_base_name}'
                        create_transparent_mask_visualization(window, base_annotations, categories, mask_path, title, create_debug=False)

                    # Add to batch buffer
                    try:
                        if output_format == 'yolo' and labels_dir:
                            batch_append_to_yolo_buffer(image_info, base_annotations)
                            flush_yolo_batch(labels_dir, annotation_type, force=False)
                        else:
                            batch_append_to_coco_buffer([image_info], base_annotations, annotation_type, coco_file_path=active_coco_file)
                            flush_coco_batch(active_coco_file, force=False)
                        images_created += 1
                        annotations_created += len(base_annotations)
                    except Exception as e:
                        print(f"        ERROR buffering window annotations: {e}")

                    current_image_id += 1

                    # Generate rotated versions of windows
                    if config.get('rotation', {}).get('enabled', False):
                        rotation_config = config['rotation']
                        rotation_angles = generate_random_angles(rotation_config['count'], rotation_config['angle_range'])

                        for rot_idx, angle in enumerate(rotation_angles):
                            # Rotate window and annotations - SIMPLE approach (no padding)
                            window_annotations_copy = copy.deepcopy(base_annotations)
                            rotated_image, rotated_annotations, (new_width, new_height) = rotate_image_and_annotations(
                                window, window_annotations_copy, angle,
                                rotation_config['interpolation'],
                                rotation_config['fill_value'],
                                defer_clipping=False  # Clip immediately to rotated bounds
                            )

                            # Keep rotated dimensions (no padding)
                            final_rot_width, final_rot_height = new_width, new_height
                            print(f"          Rotated window to {final_rot_width}x{final_rot_height}, preserved {len(rotated_annotations)}/{len(base_annotations)} annotations")

                            # Update annotation IDs (do this after clipping)
                            for ann in rotated_annotations:
                                ann['id'] = current_annotation_id
                                ann['image_id'] = current_image_id
                                current_annotation_id += 1

                            # Apply grayscale if enabled
                            rot_window_final_image = convert_to_grayscale(rotated_image) if grayscale_enabled else rotated_image

                            # Save rotated window image (no verification for speed)
                            rot_image_name = f'{file_base_name}_win_{win_idx+1}_{config_name}_rot_{rot_idx+1}.png'
                            rot_image_path = os.path.join(split_images_dir, rot_image_name)

                            success, rot_window_actual_width, rot_window_actual_height = unicode_safe_imwrite(
                                rot_image_path, cv2.cvtColor(rot_window_final_image, cv2.COLOR_RGB2BGR), verify=False
                            )

                            if not success:
                                rot_window_actual_height, rot_window_actual_width = rotated_image.shape[:2]

                            # Create rotated window image info with verified dimensions
                            rotated_image_info = {
                                "id": current_image_id,
                                "width": rot_window_actual_width,
                                "height": rot_window_actual_height,
                                "file_name": rot_image_name
                            }

                            # Create rotated visualization (skip debug for performance)
                            if config['visualization']['create_masks']:
                                rot_mask_name = f'{file_base_name}_win_{win_idx+1}_{config_name}_rot_{rot_idx+1}_masks.png'
                                rot_mask_path = os.path.join(visualizations_dir, rot_mask_name)
                                rot_title = f'{mode.title()} Window {win_idx+1} Rot {rot_idx+1} ({angle:.1f}°) - {noise_config["name"]}, {window_config["name"]} - {file_base_name}'
                                create_transparent_mask_visualization(rotated_image, rotated_annotations, categories, rot_mask_path, rot_title, create_debug=False)

                            # Add to batch buffer
                            try:
                                if output_format == 'yolo' and labels_dir:
                                    batch_append_to_yolo_buffer(rotated_image_info, rotated_annotations)
                                    flush_yolo_batch(labels_dir, annotation_type, force=False)
                                else:
                                    batch_append_to_coco_buffer([rotated_image_info], rotated_annotations, annotation_type, coco_file_path=active_coco_file)
                                    flush_coco_batch(active_coco_file, force=False)
                                images_created += 1
                                annotations_created += len(rotated_annotations)
                            except Exception as e:
                                print(f"          ERROR buffering rotated window annotations: {e}")

                            current_image_id += 1

    return images_created, annotations_created
