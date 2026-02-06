"""
Hierarchical dataset generation pipeline for H-DETR.

This module processes GIS data to generate dual-level annotations:
- Level 0 (L0): County (縣市) - 22 classes
- Level 1 (L1): Township (鄉鎮區) - 368 classes

Each image gets:
- L0 annotations (county boundaries)
- L1 annotations (township boundaries within visible counties)

Output structure:
    train/
    ├── images/
    │   └── map_001.png
    └── labels/
        ├── map_001_l0.txt    # County annotations
        └── map_001_l1.txt    # Township annotations
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any, List, Optional

from .hierarchy import (
    HierarchyManager,
    get_hierarchy_manager,
    initialize_hierarchy,
    is_hierarchical_mode,
)
from .geometry import (
    find_tif_files,
    load_data,
    crop_image_and_shapefile,
    apply_mask_regions_to_shapefile,
    generate_windows,
)
from .masks import (
    get_mask_database,
    filter_annotations_with_masks,
)
from .image import (
    preprocess_raster,
    apply_distance_based_noise,
    convert_to_grayscale,
    unicode_safe_imwrite,
    apply_hue_augmentation,
    generate_hue_augmentation_params,
    rotate_image_and_annotations,
    generate_random_angles,
)
from .annotations import (
    polygon_to_coco_segmentation,
    calculate_bbox_from_segmentation,
    calculate_area_from_segmentation,
)
from .writers.hierarchical_yolo import (
    initialize_hierarchical_yolo_dataset,
    batch_append_hierarchical,
    flush_hierarchical_batch,
    set_hierarchical_batch_size,
)
from .writers.yolo import (
    get_split_for_file,
    get_split_directories,
    initialize_split_manager,
)
from .visualization import (
    create_hierarchical_visualization,
    create_hierarchical_overlay_visualization,
)
from .core.constants import MIN_POLYGON_AREA, MIN_SEGMENTATION_LENGTH


def create_annotations_for_level(
    shapefile,
    image_shape: Tuple[int, int, int],
    transform,
    image_id: int,
    level: int,
    hierarchy_manager: HierarchyManager
) -> List[Dict[str, Any]]:
    """
    Create COCO-format annotations for a specific hierarchy level.

    Args:
        shapefile: GeoDataFrame with geometries
        image_shape: (height, width, channels)
        transform: Rasterio affine transform
        image_id: ID for the image
        level: 0 for county, 1 for township
        hierarchy_manager: HierarchyManager instance

    Returns:
        List of COCO annotation dicts
    """
    annotations = []
    height, width = image_shape[:2]

    # Determine which columns to use based on level
    if level == 0:
        id_column = hierarchy_manager.county_level.id_column
        get_class_id = hierarchy_manager.get_county_class_id
    else:
        id_column = hierarchy_manager.township_level.id_column
        get_class_id = hierarchy_manager.get_township_class_id

    for idx, row in shapefile.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        # Get entity ID and class ID
        entity_id = row.get(id_column)
        if entity_id is None:
            continue

        class_id = get_class_id(entity_id)
        if class_id is None:
            continue

        # Handle MultiPolygon and Polygon
        if geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)
        elif geom.geom_type == 'Polygon':
            polygons = [geom]
        else:
            continue

        for poly in polygons:
            if poly.is_empty or not poly.is_valid:
                continue

            # Convert geo coordinates to pixel coordinates - EXTERIOR
            exterior_coords = list(poly.exterior.coords)
            pixel_coords = []

            for x, y in exterior_coords[:-1]:  # Exclude closing point
                px, py = ~transform * (x, y)
                # Clamp to image bounds
                px = max(0, min(px, width - 1))
                py = max(0, min(py, height - 1))
                pixel_coords.extend([px, py])

            if len(pixel_coords) < 6:  # Need at least 3 points
                continue

            # Build segmentation list - exterior first, then holes
            segmentation = [pixel_coords]

            # Add interior holes (from mask subtraction)
            for interior in poly.interiors:
                hole_coords = []
                for x, y in list(interior.coords)[:-1]:  # Exclude closing point
                    px, py = ~transform * (x, y)
                    px = max(0, min(px, width - 1))
                    py = max(0, min(py, height - 1))
                    hole_coords.extend([px, py])
                if len(hole_coords) >= 6:
                    segmentation.append(hole_coords)

            # Calculate area (exterior minus holes)
            area = calculate_area_from_segmentation(pixel_coords)
            # Subtract hole areas
            for hole_seg in segmentation[1:]:
                area -= calculate_area_from_segmentation(hole_seg)

            if area < MIN_POLYGON_AREA:
                continue

            # Calculate bbox (from exterior only)
            bbox = calculate_bbox_from_segmentation(pixel_coords)

            annotation = {
                'id': len(annotations),  # Will be updated later
                'image_id': image_id,
                'category_id': class_id,
                'segmentation': segmentation,  # Now includes holes
                'bbox': bbox,
                'area': area,
                'iscrowd': 0,
                'entity_id': entity_id,  # Extra field for debugging
            }

            # Add parent info for L1
            if level == 1:
                parent_id = hierarchy_manager.get_parent_county_id(entity_id)
                parent_class = hierarchy_manager.get_parent_county_class(entity_id)
                annotation['parent_id'] = parent_id
                annotation['parent_class'] = parent_class

            annotations.append(annotation)

    return annotations


def save_hierarchical_image_and_annotations(
    image: np.ndarray,
    annotations_l0: List[Dict[str, Any]],
    annotations_l1: List[Dict[str, Any]],
    image_id: int,
    file_base: str,
    split_images_dir: str,
    split_labels_dir: str,
    grayscale_enabled: bool = False,
    viz_dir: Optional[str] = None,
    categories_l0: Optional[List[Dict]] = None,
    categories_l1: Optional[List[Dict]] = None,
    create_viz: bool = False
) -> Tuple[int, int, int]:
    """
    Save image and buffer hierarchical annotations, optionally creating visualization.

    Args:
        image: Image array (RGB)
        annotations_l0: L0 (county) annotations
        annotations_l1: L1 (township) annotations
        image_id: Current image ID
        file_base: Base filename without extension
        split_images_dir: Directory for images
        split_labels_dir: Directory for labels
        grayscale_enabled: Whether to convert to grayscale
        viz_dir: Directory for visualizations (optional)
        categories_l0: County categories for visualization (optional)
        categories_l1: Township categories for visualization (optional)
        create_viz: Whether to create visualization

    Returns:
        Tuple of (1 if success else 0, num_l0_annotations, num_l1_annotations)
    """
    # Apply grayscale if enabled
    final_image = convert_to_grayscale(image) if grayscale_enabled else image

    # Generate filename and save
    image_name = f"{file_base}.png"
    image_path = os.path.join(split_images_dir, image_name)

    success, actual_width, actual_height = unicode_safe_imwrite(
        image_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR), verify=False
    )

    if not success:
        actual_height, actual_width = image.shape[:2]

    # Update annotation image dimensions if needed
    for ann in annotations_l0:
        ann['image_id'] = image_id
    for ann in annotations_l1:
        ann['image_id'] = image_id

    # Create image info
    image_info = {
        'id': image_id,
        'width': actual_width,
        'height': actual_height,
        'file_name': image_name,
    }

    # Buffer annotations with target directory for correct split handling
    batch_append_hierarchical(image_info, annotations_l0, annotations_l1, labels_dir=split_labels_dir)

    # Flush if needed
    flush_hierarchical_batch(annotation_type='segmentation', force=False)

    # Create visualization if requested
    if create_viz and viz_dir and categories_l0 and categories_l1:
        os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, f"{file_base}_viz.png")
        try:
            create_hierarchical_overlay_visualization(
                image,  # Use original RGB image for viz
                annotations_l0,
                annotations_l1,
                categories_l0,
                categories_l1,
                viz_path,
                title=f"Hierarchical: {file_base}"
            )
        except Exception as e:
            print(f"    Warning: Failed to create visualization: {e}")

    return 1, len(annotations_l0), len(annotations_l1)


def apply_township_dropout(
    township_gdf,
    config: Dict[str, Any],
    hierarchy_manager: HierarchyManager,
    county_id: str,
    image_seed: Optional[int] = None
) -> Tuple[Any, List[str]]:
    """
    Apply township dropout to simulate incomplete maps.

    Removes selected townships from the GeoDataFrame based on configuration.

    Args:
        township_gdf: Township GeoDataFrame to modify
        config: Configuration dict containing township_dropout settings
        hierarchy_manager: HierarchyManager instance
        county_id: ID of county being processed
        image_seed: Optional seed for reproducible random dropout

    Returns:
        Tuple of (modified_township_gdf, list_of_dropped_township_ids)
    """
    dropout_config = config.get('township_dropout', {})

    if not dropout_config.get('enabled', False):
        return township_gdf, []

    mode = dropout_config.get('mode', 'random')

    # For random mode, use image_seed if provided for per-image variation
    seed = image_seed
    if seed is None:
        random_config = dropout_config.get('random', {})
        # Handle both dict and dataclass-style access
        if hasattr(random_config, 'seed'):
            seed = random_config.seed
        else:
            seed = random_config.get('seed') if random_config else None

    # Select townships to drop
    drop_ids = hierarchy_manager.select_dropout_townships(
        county_id, mode, dropout_config, seed
    )

    if not drop_ids:
        return township_gdf, []

    # Remove from township GeoDataFrame
    id_col = hierarchy_manager.township_level.id_column
    modified_gdf = township_gdf[~township_gdf[id_col].isin(drop_ids)].copy()

    return modified_gdf, drop_ids


def apply_county_holes_for_dropout(
    county_gdf,
    township_gdf,
    dropped_ids: List[str],
    hierarchy_manager: HierarchyManager
):
    """
    Subtract dropped township geometries from county geometry to create holes.

    Args:
        county_gdf: County GeoDataFrame to modify
        township_gdf: Original township GeoDataFrame (before dropout)
        dropped_ids: List of dropped township IDs
        hierarchy_manager: HierarchyManager instance

    Returns:
        Modified county GeoDataFrame with holes where dropped townships were
    """
    from shapely.ops import unary_union

    if not dropped_ids:
        return county_gdf

    id_col = hierarchy_manager.township_level.id_column

    # Get geometries of dropped townships
    dropped_townships = township_gdf[township_gdf[id_col].isin(dropped_ids)]

    if dropped_townships.empty:
        return county_gdf

    # Combine all dropped township geometries
    dropped_union = unary_union(dropped_townships.geometry.tolist())

    # Subtract from county geometry
    modified_county = county_gdf.copy()

    for idx in modified_county.index:
        geom = modified_county.loc[idx, 'geometry']
        if geom is not None and not geom.is_empty:
            # Check if this county contains any of the dropped townships
            if geom.intersects(dropped_union):
                new_geom = geom.difference(dropped_union)
                if new_geom.is_valid and not new_geom.is_empty:
                    modified_county.loc[idx, 'geometry'] = new_geom

    return modified_county


def apply_image_mask_for_dropout(
    image: np.ndarray,
    township_gdf,
    dropped_ids: List[str],
    transform,
    hierarchy_manager: HierarchyManager,
    fill_value: int = 0
) -> np.ndarray:
    """
    Mask out dropped township regions in the image itself.

    This ensures the image visually matches the labels - where townships
    are dropped, the image pixels are blanked out (filled with fill_value).

    Args:
        image: Image array (height, width, channels)
        township_gdf: Original township GeoDataFrame (before dropout)
        dropped_ids: List of dropped township IDs
        transform: Rasterio affine transform for geo->pixel conversion
        hierarchy_manager: HierarchyManager instance
        fill_value: Value to fill masked regions (0=black, 255=white)

    Returns:
        Modified image with dropped township regions masked
    """
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

    if not dropped_ids:
        return image

    id_col = hierarchy_manager.township_level.id_column

    # Get geometries of dropped townships
    dropped_townships = township_gdf[township_gdf[id_col].isin(dropped_ids)]

    if dropped_townships.empty:
        return image

    height, width = image.shape[:2]

    # Create mask (1 = keep, 0 = drop)
    mask = np.ones((height, width), dtype=np.uint8)

    for _, row in dropped_townships.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        # Handle MultiPolygon and Polygon
        if geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)
        elif geom.geom_type == 'Polygon':
            polygons = [geom]
        else:
            continue

        for poly in polygons:
            if poly.is_empty or not poly.is_valid:
                continue

            # Convert geo coordinates to pixel coordinates
            exterior_coords = list(poly.exterior.coords)
            pixel_coords = []

            for x, y in exterior_coords:
                px, py = ~transform * (x, y)
                pixel_coords.append([int(px), int(py)])

            if len(pixel_coords) >= 3:
                pts = np.array(pixel_coords, dtype=np.int32)
                # Fill the dropped township region with 0 in mask
                cv2.fillPoly(mask, [pts], 0)

                # Handle interior holes (islands within the township)
                # These should NOT be masked out
                for interior in poly.interiors:
                    hole_coords = list(interior.coords)
                    pixel_hole = []
                    for x, y in hole_coords:
                        px, py = ~transform * (x, y)
                        pixel_hole.append([int(px), int(py)])

                    if len(pixel_hole) >= 3:
                        hole_pts = np.array(pixel_hole, dtype=np.int32)
                        # Fill hole back to 1 (keep these pixels)
                        cv2.fillPoly(mask, [hole_pts], 1)

    # Apply mask to image
    masked_image = image.copy()

    if len(image.shape) == 3:
        # Color image - apply mask to all channels
        for c in range(image.shape[2]):
            masked_image[:, :, c] = np.where(mask == 1, image[:, :, c], fill_value)
    else:
        # Grayscale image
        masked_image = np.where(mask == 1, image, fill_value)

    return masked_image


def rotate_hierarchical_image_and_annotations(
    image: np.ndarray,
    annotations_l0: List[Dict[str, Any]],
    annotations_l1: List[Dict[str, Any]],
    angle: float,
    rotation_config: Dict[str, Any]
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], Tuple[int, int]]:
    """
    Rotate image and transform both L0 and L1 annotations.

    Args:
        image: Image array
        annotations_l0: L0 (county) annotations
        annotations_l1: L1 (township) annotations
        angle: Rotation angle in degrees
        rotation_config: Rotation configuration dict

    Returns:
        Tuple of (rotated_image, rotated_l0, rotated_l1, (new_width, new_height))
    """
    import copy

    # Deep copy annotations to avoid modifying originals
    l0_copy = copy.deepcopy(annotations_l0)
    l1_copy = copy.deepcopy(annotations_l1)

    # Rotate image with L0 annotations
    rotated_image, rotated_l0, new_dims = rotate_image_and_annotations(
        image,
        l0_copy,
        angle,
        interpolation=rotation_config.get('interpolation', 'bilinear'),
        fill_value=rotation_config.get('fill_value', 0),
        defer_clipping=False
    )

    # Rotate L1 annotations separately using same parameters
    # We need to recreate the rotation for L1 annotations
    _, rotated_l1, _ = rotate_image_and_annotations(
        image,  # Use original image for consistent transform
        l1_copy,
        angle,
        interpolation=rotation_config.get('interpolation', 'bilinear'),
        fill_value=rotation_config.get('fill_value', 0),
        defer_clipping=False
    )

    return rotated_image, rotated_l0, rotated_l1, new_dims


def process_hierarchical_separate_districts(
    config: Dict[str, Any],
    base_output_dir: str
) -> Tuple[int, int, int]:
    """
    Process districts with hierarchical (dual-level) annotations.

    Each image contains ONE county region, with:
    - L0: The county boundary
    - L1: All townships within that county

    Args:
        config: Configuration dictionary
        base_output_dir: Output directory path

    Returns:
        Tuple of (images_created, l0_annotations, l1_annotations)
    """
    print("\n=== HIERARCHICAL PROCESSING: SEPARATE DISTRICTS ===")
    print("Each image contains ONE county with its townships")

    # Initialize hierarchy manager
    hierarchy_manager = get_hierarchy_manager()
    if not hierarchy_manager.loaded:
        if not initialize_hierarchy(config):
            print("ERROR: Failed to initialize hierarchy")
            return 0, 0, 0

    # Initialize dataset structure
    categories_l0 = hierarchy_manager.create_county_categories()
    categories_l1 = hierarchy_manager.create_township_categories()

    hierarchy_file = config.get('hierarchy', {}).get('hierarchy_file', '')
    use_split = config.get('output', {}).get('use_split', True)

    initialize_hierarchical_yolo_dataset(
        base_output_dir,
        categories_l0,
        categories_l1,
        hierarchy_file=hierarchy_file,
        use_split=use_split
    )

    # Set batch size
    batch_size = config.get('performance', {}).get('batch_size', 10)
    set_hierarchical_batch_size(batch_size)

    # Initialize split manager
    if use_split:
        train_ratio = config.get('output', {}).get('train_ratio', 0.7)
        val_ratio = config.get('output', {}).get('val_ratio', 0.2)
        test_ratio = config.get('output', {}).get('test_ratio', 0.1)
        split_seed = config.get('output', {}).get('split_seed', 42)

        # Collect all TIF files
        all_tif_files = []
        mapdata_base_dir = config.get('mapdata_base_dir', 'datasets/MAPDATA')
        for district in config['districts']:
            tif_files = find_tif_files(district, mapdata_base_dir)
            max_files = config['processing'].get('max_files_per_district')
            if max_files and len(tif_files) > max_files:
                tif_files = tif_files[:max_files]
            all_tif_files.extend(tif_files)

        initialize_split_manager(
            base_output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
            file_list=all_tif_files
        )

    # Get shapefiles
    county_gdf = hierarchy_manager.get_county_gdf()
    township_gdf = hierarchy_manager.get_township_gdf()

    # Stats
    total_images = 0
    total_l0_annotations = 0
    total_l1_annotations = 0
    image_id = 1

    # Config settings
    crop_factor = config.get('crop_factor', 0.05)
    grayscale_enabled = config.get('output', {}).get('grayscale', False)
    create_viz = config.get('visualization', {}).get('create_masks', True)
    create_hier_overlay = config.get('visualization', {}).get('create_hierarchical_overlay', True)

    # Process each district
    mapdata_base_dir = config.get('mapdata_base_dir', 'datasets/MAPDATA')

    for district in tqdm(config['districts'], desc="Districts"):
        tif_files = find_tif_files(district, mapdata_base_dir)

        max_files = config['processing'].get('max_files_per_district')
        if max_files and len(tif_files) > max_files:
            tif_files = tif_files[:max_files]

        # Get county ID for this district (for dropout)
        district_county_id = hierarchy_manager.get_county_id_by_district_name(district)
        if district_county_id:
            tqdm.write(f"  District '{district}' -> County ID '{district_county_id}'")

        for tif_file in tqdm(tif_files, desc=f"  {district}", leave=False):
            try:
                # Load raster with COUNTY shapefile
                county_shapefile_path = config.get('hierarchy', {}).get('level_0', {}).get(
                    'shapefile', config.get('shapefile_path')
                )
                raster, county_reprojected = load_data(tif_file, county_shapefile_path)

                if raster is None or county_reprojected is None:
                    continue

                # Also load township shapefile
                township_shapefile_path = config.get('hierarchy', {}).get('level_1', {}).get('shapefile')
                _, township_reprojected = load_data(tif_file, township_shapefile_path)

                if township_reprojected is None:
                    raster.close()
                    continue

                # Load mask regions from database
                tif_basename = os.path.basename(tif_file)
                mask_db = get_mask_database()
                mask_regions = mask_db.get_masks(tif_basename)
                raster_transform = raster.transform  # Save for mask application

                if mask_regions:
                    tqdm.write(f"      Found {len(mask_regions)} mask region(s) for {tif_basename}")

                    # Check mask coverage threshold
                    from shapely.geometry import box as shapely_box
                    from shapely.ops import unary_union

                    # Get district bounds in geo coords
                    district_bounds = county_reprojected.total_bounds
                    district_image_bounds_geo = shapely_box(*district_bounds)
                    district_area = district_image_bounds_geo.area

                    # Convert mask regions to geographic polygons
                    mask_polygons_geo = []
                    for region in mask_regions:
                        x1_px, y1_px = region['x'], region['y']
                        x2_px, y2_px = x1_px + region['width'], y1_px + region['height']
                        x1_geo, y1_geo = raster_transform * (x1_px, y1_px)
                        x2_geo, y2_geo = raster_transform * (x2_px, y2_px)
                        mask_poly = shapely_box(
                            min(x1_geo, x2_geo), min(y1_geo, y2_geo),
                            max(x1_geo, x2_geo), max(y1_geo, y2_geo)
                        )
                        mask_polygons_geo.append(mask_poly)

                    combined_mask = unary_union(mask_polygons_geo)
                    mask_intersection_area = district_image_bounds_geo.intersection(combined_mask).area
                    coverage_pct = (mask_intersection_area / district_area * 100.0) if district_area > 0 else 0.0

                    tqdm.write(f"      Mask coverage: {coverage_pct:.1f}%")

                    # Check threshold
                    mask_skip_threshold = config.get('processing', {}).get('mask_skip_threshold_separate', 50.0)
                    if coverage_pct > mask_skip_threshold:
                        tqdm.write(f"      SKIPPED - {coverage_pct:.1f}% masked (threshold: {mask_skip_threshold}%)")
                        raster.close()
                        continue

                    # Apply masks to BOTH county and township shapefiles
                    # Save original township in case mask application fails
                    original_township = township_reprojected.copy()

                    county_reprojected = apply_mask_regions_to_shapefile(
                        county_reprojected, mask_regions, raster_transform
                    )
                    if county_reprojected is None or county_reprojected.empty:
                        tqdm.write(f"      SKIPPED - all county areas masked")
                        raster.close()
                        continue

                    masked_township = apply_mask_regions_to_shapefile(
                        township_reprojected, mask_regions, raster_transform
                    )
                    # Keep original if mask application fails
                    township_reprojected = masked_township if masked_township is not None else original_township

                # Preprocess raster
                raster_data = raster.read()
                image_data = preprocess_raster(raster_data)

                if image_data is None:
                    raster.close()
                    continue

                # Crop
                cropped_result = crop_image_and_shapefile(
                    raster, image_data, county_reprojected, crop_factor
                )

                if any(x is None for x in cropped_result):
                    raster.close()
                    continue

                cropped_image, cropped_transform, cropped_bounds, clipped_county = cropped_result

                # Also clip township shapefile to same bounds
                from shapely.geometry import box
                bounds_poly = box(*cropped_bounds)
                clipped_township = township_reprojected[
                    township_reprojected.geometry.intersects(bounds_poly)
                ].copy()

                if clipped_county.empty:
                    raster.close()
                    continue

                # Store original (uncut) versions for dropout variants
                original_clipped_county = clipped_county.copy()
                original_clipped_township = clipped_township.copy()

                # Get dropout configuration
                dropout_config = config.get('township_dropout', {})
                dropout_enabled = dropout_config.get('enabled', False) and is_hierarchical_mode(config)
                include_uncut = dropout_config.get('include_uncut', True)
                dropout_variants = dropout_config.get('dropout_variants', 1)
                create_county_holes = dropout_config.get('create_county_holes', False)

                # Build list of variants to generate: [(county_gdf, township_gdf, variant_suffix, dropped_ids), ...]
                # dropped_ids is used to mask the image where townships were dropped
                variants_to_process = []

                if not dropout_enabled or not district_county_id:
                    # No dropout - just process the original (no dropped_ids)
                    variants_to_process.append((clipped_county, clipped_township, '', []))
                else:
                    # Generate uncut version first if enabled (no dropped_ids for uncut)
                    if include_uncut:
                        variants_to_process.append((original_clipped_county.copy(),
                                                   original_clipped_township.copy(),
                                                   '_uncut', []))

                    # Generate dropout variants
                    for variant_idx in range(dropout_variants):
                        # Use different seed for each variant
                        variant_seed = image_id * 1000 + variant_idx

                        # Apply dropout to township
                        variant_township, dropped_ids = apply_township_dropout(
                            original_clipped_township.copy(), config, hierarchy_manager,
                            district_county_id, image_seed=variant_seed
                        )

                        if dropped_ids:
                            tqdm.write(f"      Variant {variant_idx}: Dropped {len(dropped_ids)} townships")

                            # Apply county holes if enabled
                            if create_county_holes:
                                variant_county = apply_county_holes_for_dropout(
                                    original_clipped_county.copy(),
                                    original_clipped_township,
                                    dropped_ids,
                                    hierarchy_manager
                                )
                            else:
                                variant_county = original_clipped_county.copy()

                            # Include dropped_ids for image masking
                            variants_to_process.append((variant_county, variant_township,
                                                       f'_drop{variant_idx}', dropped_ids))
                        else:
                            # No townships dropped - use original
                            variants_to_process.append((original_clipped_county.copy(),
                                                       original_clipped_township.copy(),
                                                       f'_drop{variant_idx}', []))

                # Get split for this file
                if use_split:
                    split_name = get_split_for_file(tif_file)
                    split_images_dir, split_labels_dir = get_split_directories(
                        base_output_dir, split_name, format_type='yolo'
                    )
                    viz_dir = os.path.join(base_output_dir, split_name, 'visualizations')
                else:
                    split_images_dir = os.path.join(base_output_dir, 'images')
                    split_labels_dir = os.path.join(base_output_dir, 'labels')
                    viz_dir = os.path.join(base_output_dir, 'visualizations')

                os.makedirs(viz_dir, exist_ok=True)

                # Process each noise configuration
                noise_configs = [nc for nc in config['noise_configs'] if nc.get('enabled', True)]

                # For val/test, only use clean (no augmentations)
                is_train = not use_split or split_name == 'train'
                if use_split and split_name in ['val', 'test']:
                    noise_configs = [nc for nc in noise_configs if nc['name'] == 'clean']
                    if not noise_configs:
                        noise_configs = [{'name': 'clean', 'intensity': 0.0, 'type': 'gaussian',
                                         'acceleration': 0, 'border_buffer_pixels': 0, 'enabled': True}]

                # Get augmentation configs
                hue_config = config.get('hue_augmentation', {})
                hue_enabled = hue_config.get('enabled', False) and is_train
                rotation_config = config.get('rotation', {})
                rotation_enabled = rotation_config.get('enabled', False) and is_train

                # Get noise target for hierarchical mode
                noise_target = config.get('hierarchy', {}).get('noise_target', 'county')

                # Process each dropout variant (uncut + cut versions)
                for variant_county, variant_township, variant_suffix, variant_dropped_ids in variants_to_process:

                    # Apply image masking for dropped townships (so image matches labels)
                    if variant_dropped_ids and create_county_holes:
                        base_image_for_variant = apply_image_mask_for_dropout(
                            cropped_image,
                            original_clipped_township,  # Use original to get dropped geometries
                            variant_dropped_ids,
                            cropped_transform,
                            hierarchy_manager,
                            fill_value=255  # White fill for dropped regions
                        )
                    else:
                        base_image_for_variant = cropped_image

                    for noise_config in noise_configs:
                        # Apply noise based on noise_target configuration
                        if noise_config['name'] != 'clean' and noise_target != 'disabled':
                            if noise_target == 'township':
                                # Apply noise based on township boundaries
                                noisy_image, _ = apply_distance_based_noise(
                                    base_image_for_variant, variant_township, cropped_transform, noise_config
                                )
                            else:  # 'county' (default)
                                # Apply noise based on county boundaries
                                noisy_image, _ = apply_distance_based_noise(
                                    base_image_for_variant, variant_county, cropped_transform, noise_config
                                )
                        else:
                            noisy_image = base_image_for_variant

                        # Create base annotations for both levels using variant shapefiles
                        base_annotations_l0 = create_annotations_for_level(
                            variant_county, noisy_image.shape, cropped_transform,
                            image_id, level=0, hierarchy_manager=hierarchy_manager
                        )

                        base_annotations_l1 = create_annotations_for_level(
                            variant_township, noisy_image.shape, cropped_transform,
                            image_id, level=1, hierarchy_manager=hierarchy_manager
                        )

                        # ============================================================
                        # SAVE BASE IMAGE (with noise, no other augmentation)
                        # ============================================================
                        file_base = f"hier_{Path(tif_file).stem}{variant_suffix}_noise_{noise_config['name']}"

                        imgs, l0_count, l1_count = save_hierarchical_image_and_annotations(
                            noisy_image, base_annotations_l0, base_annotations_l1,
                            image_id, file_base, split_images_dir, split_labels_dir,
                            grayscale_enabled,
                            viz_dir=viz_dir,
                            categories_l0=categories_l0,
                            categories_l1=categories_l1,
                            create_viz=create_viz and create_hier_overlay
                        )
                        total_images += imgs
                        total_l0_annotations += l0_count
                        total_l1_annotations += l1_count

                        image_id += 1

                        # ============================================================
                        # HUE AUGMENTATION
                        # ============================================================
                        if hue_enabled:
                            hue_params = generate_hue_augmentation_params(
                                count=hue_config.get('count', 1),
                                hue_range=hue_config.get('hue_shift_range', [-0.1, 0.1]),
                                sat_range=hue_config.get('saturation_range', [0.8, 1.2]),
                                val_range=hue_config.get('value_range', [0.9, 1.1])
                            )

                            for hue_idx, (hue_shift, sat_factor, val_factor) in enumerate(hue_params):
                                # Apply hue augmentation
                                hue_image = apply_hue_augmentation(
                                    noisy_image, hue_shift, sat_factor, val_factor
                                )

                                # Annotations stay the same (hue doesn't change geometry)
                                import copy
                                hue_annotations_l0 = copy.deepcopy(base_annotations_l0)
                                hue_annotations_l1 = copy.deepcopy(base_annotations_l1)

                                # Save hue-augmented image
                                hue_file_base = f"{file_base}_hue{hue_idx}"

                                imgs, l0_count, l1_count = save_hierarchical_image_and_annotations(
                                    hue_image, hue_annotations_l0, hue_annotations_l1,
                                    image_id, hue_file_base, split_images_dir, split_labels_dir,
                                    grayscale_enabled,
                                    viz_dir=viz_dir,
                                    categories_l0=categories_l0,
                                    categories_l1=categories_l1,
                                    create_viz=create_viz and create_hier_overlay
                                )
                                total_images += imgs
                                total_l0_annotations += l0_count
                                total_l1_annotations += l1_count
                                image_id += 1

                                # Apply rotation to hue-augmented image
                                if rotation_enabled:
                                    rotation_angles = generate_random_angles(
                                        rotation_config.get('count', 1),
                                        rotation_config.get('angle_range', [-15, 15])
                                    )

                                    for rot_idx, angle in enumerate(rotation_angles):
                                        rotated_image, rotated_l0, rotated_l1, _ = \
                                            rotate_hierarchical_image_and_annotations(
                                                hue_image, hue_annotations_l0, hue_annotations_l1,
                                                angle, rotation_config
                                            )

                                        rot_file_base = f"{hue_file_base}_rot{rot_idx}"

                                        imgs, l0_count, l1_count = save_hierarchical_image_and_annotations(
                                            rotated_image, rotated_l0, rotated_l1,
                                            image_id, rot_file_base, split_images_dir, split_labels_dir,
                                            grayscale_enabled,
                                            viz_dir=viz_dir,
                                            categories_l0=categories_l0,
                                            categories_l1=categories_l1,
                                            create_viz=create_viz and create_hier_overlay
                                        )
                                        total_images += imgs
                                        total_l0_annotations += l0_count
                                        total_l1_annotations += l1_count
                                        image_id += 1

                        # ============================================================
                        # ROTATION AUGMENTATION (on base noisy image, without hue)
                        # ============================================================
                        if rotation_enabled:
                            rotation_angles = generate_random_angles(
                                rotation_config.get('count', 1),
                                rotation_config.get('angle_range', [-15, 15])
                            )

                            for rot_idx, angle in enumerate(rotation_angles):
                                rotated_image, rotated_l0, rotated_l1, _ = \
                                    rotate_hierarchical_image_and_annotations(
                                        noisy_image, base_annotations_l0, base_annotations_l1,
                                        angle, rotation_config
                                    )

                                rot_file_base = f"{file_base}_rot{rot_idx}"

                                imgs, l0_count, l1_count = save_hierarchical_image_and_annotations(
                                    rotated_image, rotated_l0, rotated_l1,
                                    image_id, rot_file_base, split_images_dir, split_labels_dir,
                                    grayscale_enabled,
                                    viz_dir=viz_dir,
                                    categories_l0=categories_l0,
                                    categories_l1=categories_l1,
                                    create_viz=create_viz and create_hier_overlay
                                )
                                total_images += imgs
                                total_l0_annotations += l0_count
                                total_l1_annotations += l1_count
                                image_id += 1

                raster.close()

            except Exception as e:
                print(f"    Error processing {tif_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Final flush - each buffered entry has its own labels_dir stored
    # so a single flush will write each entry to its correct split directory
    flush_hierarchical_batch(annotation_type='segmentation', force=True)

    print(f"\nHierarchical processing complete:")
    print(f"  Images: {total_images}")
    print(f"  L0 (County) annotations: {total_l0_annotations}")
    print(f"  L1 (Township) annotations: {total_l1_annotations}")

    return total_images, total_l0_annotations, total_l1_annotations


def create_hierarchical_annotations_for_window(
    county_shapefile,
    township_shapefile,
    window_info: Dict[str, Any],
    transform,
    hierarchy_manager: HierarchyManager
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create L0 and L1 annotations for a specific window.

    Clips both county and township geometries to window bounds.
    Handles partial entities at window edges.

    Args:
        county_shapefile: GeoDataFrame with county geometries
        township_shapefile: GeoDataFrame with township geometries
        window_info: Window bounds and dimensions
        transform: Affine transform for coordinate conversion
        hierarchy_manager: HierarchyManager instance

    Returns:
        Tuple of (l0_annotations, l1_annotations)
    """
    from shapely.geometry import Polygon as ShapelyPolygon, box

    annotations_l0 = []
    annotations_l1 = []

    # Window bounds in pixel coordinates
    start_x = window_info['start_x']
    start_y = window_info['start_y']
    end_x = window_info['end_x']
    end_y = window_info['end_y']
    window_width = end_x - start_x
    window_height = end_y - start_y

    window_box = box(start_x, start_y, end_x, end_y)

    def process_shapefile_for_window(shapefile, level: int):
        """Process a shapefile and create window-clipped annotations."""
        annotations = []

        if level == 0:
            id_column = hierarchy_manager.county_level.id_column
            get_class_id = hierarchy_manager.get_county_class_id
        else:
            id_column = hierarchy_manager.township_level.id_column
            get_class_id = hierarchy_manager.get_township_class_id

        for idx, row in shapefile.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            entity_id = row.get(id_column)
            if entity_id is None:
                continue

            class_id = get_class_id(entity_id)
            if class_id is None:
                continue

            # Handle MultiPolygon and Polygon
            if geom.geom_type == 'MultiPolygon':
                polygons = list(geom.geoms)
            elif geom.geom_type == 'Polygon':
                polygons = [geom]
            else:
                continue

            for poly in polygons:
                if poly.is_empty or not poly.is_valid:
                    continue

                # Convert exterior to pixel coordinates
                exterior_coords = list(poly.exterior.coords)
                pixel_exterior = []
                for x, y in exterior_coords[:-1]:
                    px, py = ~transform * (x, y)
                    pixel_exterior.append([px, py])

                if len(pixel_exterior) < 3:
                    continue

                # Convert interior holes to pixel coordinates (IMPORTANT for masks!)
                pixel_interiors = []
                for interior in poly.interiors:
                    hole_coords = list(interior.coords)
                    pixel_hole = []
                    for x, y in hole_coords[:-1]:
                        px, py = ~transform * (x, y)
                        pixel_hole.append([px, py])
                    if len(pixel_hole) >= 3:
                        pixel_interiors.append(pixel_hole)

                # Create polygon with holes preserved
                pixel_polygon = ShapelyPolygon(pixel_exterior, pixel_interiors)

                # Clip to window bounds
                clipped = pixel_polygon.intersection(window_box)

                if clipped.is_empty:
                    continue

                # Handle result (may be Polygon or MultiPolygon)
                if clipped.geom_type == 'MultiPolygon':
                    clipped_polys = list(clipped.geoms)
                elif clipped.geom_type == 'Polygon':
                    clipped_polys = [clipped]
                else:
                    continue

                for clipped_poly in clipped_polys:
                    if clipped_poly.is_empty or clipped_poly.area < MIN_POLYGON_AREA:
                        continue

                    # Convert to window-relative coordinates
                    seg_coords = []
                    for x, y in list(clipped_poly.exterior.coords)[:-1]:
                        # Translate to window origin
                        wx = max(0, min(x - start_x, window_width - 1))
                        wy = max(0, min(y - start_y, window_height - 1))
                        seg_coords.extend([wx, wy])

                    if len(seg_coords) < 6:
                        continue

                    segmentation = [seg_coords]

                    # Handle holes
                    for interior in clipped_poly.interiors:
                        hole_coords = []
                        for x, y in list(interior.coords)[:-1]:
                            wx = max(0, min(x - start_x, window_width - 1))
                            wy = max(0, min(y - start_y, window_height - 1))
                            hole_coords.extend([wx, wy])
                        if len(hole_coords) >= 6:
                            segmentation.append(hole_coords)

                    area = calculate_area_from_segmentation(seg_coords)
                    for hole_seg in segmentation[1:]:
                        area -= calculate_area_from_segmentation(hole_seg)

                    if area < MIN_POLYGON_AREA:
                        continue

                    bbox = calculate_bbox_from_segmentation(seg_coords)

                    annotation = {
                        'id': len(annotations),
                        'image_id': 0,  # Will be updated later
                        'category_id': class_id,
                        'segmentation': segmentation,
                        'bbox': bbox,
                        'area': area,
                        'iscrowd': 0,
                        'entity_id': entity_id,
                    }

                    if level == 1:
                        parent_id = hierarchy_manager.get_parent_county_id(entity_id)
                        parent_class = hierarchy_manager.get_parent_county_class(entity_id)
                        annotation['parent_id'] = parent_id
                        annotation['parent_class'] = parent_class

                    annotations.append(annotation)

        return annotations

    # Process both levels
    annotations_l0 = process_shapefile_for_window(county_shapefile, level=0)
    annotations_l1 = process_shapefile_for_window(township_shapefile, level=1)

    return annotations_l0, annotations_l1


def process_hierarchical_combined_maps(
    config: Dict[str, Any],
    base_output_dir: str
) -> Tuple[int, int, int]:
    """
    Process combined maps with hierarchical (dual-level) annotations.

    Each image contains ALL counties in the map, with:
    - L0: All county boundaries visible in the image
    - L1: All townships within visible counties

    Supports window_configs for sliding window crops.

    Args:
        config: Configuration dictionary
        base_output_dir: Output directory path

    Returns:
        Tuple of (images_created, l0_annotations, l1_annotations)
    """
    print("\n=== HIERARCHICAL PROCESSING: COMBINED MAPS ===")
    print("Each image contains ALL counties with their townships")

    # Initialize hierarchy manager
    hierarchy_manager = get_hierarchy_manager()
    if not hierarchy_manager.loaded:
        if not initialize_hierarchy(config):
            print("ERROR: Failed to initialize hierarchy")
            return 0, 0, 0

    # Get categories
    categories_l0 = hierarchy_manager.create_county_categories()
    categories_l1 = hierarchy_manager.create_township_categories()

    use_split = config.get('output', {}).get('use_split', True)

    # Stats
    total_images = 0
    total_l0_annotations = 0
    total_l1_annotations = 0
    image_id = 1

    # Config settings
    crop_factor = config.get('crop_factor', 0.05)
    grayscale_enabled = config.get('output', {}).get('grayscale', False)
    create_viz = config.get('visualization', {}).get('create_masks', True)
    create_hier_overlay = config.get('visualization', {}).get('create_hierarchical_overlay', True)

    # Collect all TIF files
    mapdata_base_dir = config.get('mapdata_base_dir', 'datasets/MAPDATA')
    all_tif_files = []
    for district in config['districts']:
        tif_files = find_tif_files(district, mapdata_base_dir)
        max_files = config['processing'].get('max_files_per_district')
        if max_files and len(tif_files) > max_files:
            tif_files = tif_files[:max_files]
        all_tif_files.extend(tif_files)

    # Initialize split manager if needed
    if use_split:
        train_ratio = config.get('output', {}).get('train_ratio', 0.7)
        val_ratio = config.get('output', {}).get('val_ratio', 0.2)
        test_ratio = config.get('output', {}).get('test_ratio', 0.1)
        split_seed = config.get('output', {}).get('split_seed', 42)

        initialize_split_manager(
            base_output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
            file_list=all_tif_files
        )

    print(f"Processing {len(all_tif_files)} TIF files in combined mode")

    for tif_file in tqdm(all_tif_files, desc="Combined Maps"):
        try:
            # Load both shapefiles
            county_shapefile_path = config.get('hierarchy', {}).get('level_0', {}).get(
                'shapefile', config.get('shapefile_path')
            )
            township_shapefile_path = config.get('hierarchy', {}).get('level_1', {}).get('shapefile')

            raster, county_reprojected = load_data(tif_file, county_shapefile_path)
            if raster is None or county_reprojected is None:
                continue

            _, township_reprojected = load_data(tif_file, township_shapefile_path)
            if township_reprojected is None:
                raster.close()
                continue

            # Load mask regions from database
            tif_basename = os.path.basename(tif_file)
            mask_db = get_mask_database()
            mask_regions = mask_db.get_masks(tif_basename)
            raster_transform = raster.transform

            if mask_regions:
                tqdm.write(f"    Found {len(mask_regions)} mask region(s) for {tif_basename}")

                # Apply masks to BOTH county and township shapefiles
                # Save original township in case mask application fails
                original_township = township_reprojected.copy()

                county_reprojected = apply_mask_regions_to_shapefile(
                    county_reprojected, mask_regions, raster_transform
                )
                if county_reprojected is None or county_reprojected.empty:
                    tqdm.write(f"    SKIPPED - all county areas masked")
                    raster.close()
                    continue

                masked_township = apply_mask_regions_to_shapefile(
                    township_reprojected, mask_regions, raster_transform
                )
                # Keep original if mask application fails
                township_reprojected = masked_township if masked_township is not None else original_township

            # Preprocess raster
            raster_data = raster.read()
            image_data = preprocess_raster(raster_data)

            if image_data is None:
                raster.close()
                continue

            # Crop (keep all counties)
            cropped_result = crop_image_and_shapefile(
                raster, image_data, county_reprojected, crop_factor
            )

            if any(x is None for x in cropped_result):
                raster.close()
                continue

            cropped_image, cropped_transform, cropped_bounds, clipped_county = cropped_result

            # Clip township to same bounds
            from shapely.geometry import box
            bounds_poly = box(*cropped_bounds)
            clipped_township = township_reprojected[
                township_reprojected.geometry.intersects(bounds_poly)
            ].copy()

            if clipped_county.empty:
                raster.close()
                continue

            # Get split directories
            if use_split:
                split_name = get_split_for_file(tif_file)
                split_images_dir, split_labels_dir = get_split_directories(
                    base_output_dir, split_name, format_type='yolo'
                )
                viz_dir = os.path.join(base_output_dir, split_name, 'visualizations')
            else:
                split_images_dir = os.path.join(base_output_dir, 'images')
                split_labels_dir = os.path.join(base_output_dir, 'labels')
                viz_dir = os.path.join(base_output_dir, 'visualizations')

            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)

            is_train = not use_split or split_name == 'train'

            # Get augmentation configs
            hue_config = config.get('hue_augmentation', {})
            hue_enabled = hue_config.get('enabled', False) and is_train
            rotation_config = config.get('rotation', {})
            rotation_enabled = rotation_config.get('enabled', False) and is_train
            noise_target = config.get('hierarchy', {}).get('noise_target', 'county')

            # Get noise configs
            noise_configs = [nc for nc in config.get('noise_configs', []) if nc.get('enabled', True)]
            if use_split and split_name in ['val', 'test']:
                noise_configs = [nc for nc in noise_configs if nc['name'] == 'clean']
                if not noise_configs:
                    noise_configs = [{'name': 'clean', 'intensity': 0.0, 'type': 'gaussian',
                                     'acceleration': 0, 'border_buffer_pixels': 0, 'enabled': True}]

            # Process window configurations
            window_configs = config.get('window_configs', [])
            if not window_configs:
                # If no window configs, create one for the full image
                window_configs = [{'name': 'full', 'x_percent': 100, 'y_percent': 100}]

            for noise_config in noise_configs:
                # Apply noise
                if noise_config['name'] != 'clean' and noise_target != 'disabled':
                    if noise_target == 'township':
                        noisy_image, _ = apply_distance_based_noise(
                            cropped_image, clipped_township, cropped_transform, noise_config
                        )
                    else:
                        noisy_image, _ = apply_distance_based_noise(
                            cropped_image, clipped_county, cropped_transform, noise_config
                        )
                else:
                    noisy_image = cropped_image

                # Process each window configuration
                for window_config in window_configs:
                    windows, window_infos = generate_windows(noisy_image, window_config)

                    for win_idx, (window, w_info) in enumerate(zip(windows, window_infos)):
                        # Create hierarchical annotations for this window
                        window_annotations_l0, window_annotations_l1 = \
                            create_hierarchical_annotations_for_window(
                                clipped_county, clipped_township,
                                w_info, cropped_transform, hierarchy_manager
                            )

                        # Skip if no annotations
                        if not window_annotations_l0 and not window_annotations_l1:
                            continue

                        # Generate filename
                        file_base = f"hier_combined_{Path(tif_file).stem}_noise_{noise_config['name']}_win{win_idx}_{window_config['name']}"

                        imgs, l0_count, l1_count = save_hierarchical_image_and_annotations(
                            window, window_annotations_l0, window_annotations_l1,
                            image_id, file_base, split_images_dir, split_labels_dir,
                            grayscale_enabled,
                            viz_dir=viz_dir,
                            categories_l0=categories_l0,
                            categories_l1=categories_l1,
                            create_viz=create_viz and create_hier_overlay
                        )
                        total_images += imgs
                        total_l0_annotations += l0_count
                        total_l1_annotations += l1_count
                        image_id += 1

                        # Apply hue augmentation to windows
                        if hue_enabled:
                            hue_params = generate_hue_augmentation_params(
                                count=hue_config.get('count', 1),
                                hue_range=hue_config.get('hue_shift_range', [-0.1, 0.1]),
                                sat_range=hue_config.get('saturation_range', [0.8, 1.2]),
                                val_range=hue_config.get('value_range', [0.9, 1.1])
                            )

                            for hue_idx, (hue_shift, sat_factor, val_factor) in enumerate(hue_params):
                                hue_window = apply_hue_augmentation(window, hue_shift, sat_factor, val_factor)

                                import copy
                                hue_l0 = copy.deepcopy(window_annotations_l0)
                                hue_l1 = copy.deepcopy(window_annotations_l1)

                                hue_file_base = f"{file_base}_hue{hue_idx}"
                                imgs, l0_count, l1_count = save_hierarchical_image_and_annotations(
                                    hue_window, hue_l0, hue_l1,
                                    image_id, hue_file_base, split_images_dir, split_labels_dir,
                                    grayscale_enabled,
                                    viz_dir=viz_dir,
                                    categories_l0=categories_l0,
                                    categories_l1=categories_l1,
                                    create_viz=create_viz and create_hier_overlay
                                )
                                total_images += imgs
                                total_l0_annotations += l0_count
                                total_l1_annotations += l1_count
                                image_id += 1

                        # Apply rotation to windows
                        if rotation_enabled:
                            rotation_angles = generate_random_angles(
                                rotation_config.get('count', 1),
                                rotation_config.get('angle_range', [-15, 15])
                            )

                            for rot_idx, angle in enumerate(rotation_angles):
                                rotated_window, rotated_l0, rotated_l1, _ = \
                                    rotate_hierarchical_image_and_annotations(
                                        window, window_annotations_l0, window_annotations_l1,
                                        angle, rotation_config
                                    )

                                rot_file_base = f"{file_base}_rot{rot_idx}"
                                imgs, l0_count, l1_count = save_hierarchical_image_and_annotations(
                                    rotated_window, rotated_l0, rotated_l1,
                                    image_id, rot_file_base, split_images_dir, split_labels_dir,
                                    grayscale_enabled,
                                    viz_dir=viz_dir,
                                    categories_l0=categories_l0,
                                    categories_l1=categories_l1,
                                    create_viz=create_viz and create_hier_overlay
                                )
                                total_images += imgs
                                total_l0_annotations += l0_count
                                total_l1_annotations += l1_count
                                image_id += 1

            raster.close()

        except Exception as e:
            print(f"  Error processing {tif_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final flush - each buffered entry has its own labels_dir stored
    # so a single flush will write each entry to its correct split directory
    flush_hierarchical_batch(annotation_type='segmentation', force=True)

    print(f"\nCombined maps processing complete:")
    print(f"  Images: {total_images}")
    print(f"  L0 (County) annotations: {total_l0_annotations}")
    print(f"  L1 (Township) annotations: {total_l1_annotations}")

    return total_images, total_l0_annotations, total_l1_annotations


def run_hierarchical_pipeline(config_path: str = None, config: Dict = None) -> Tuple[int, int, int]:
    """
    Run the hierarchical dataset generation pipeline.

    Args:
        config_path: Path to config file (optional if config dict provided)
        config: Configuration dictionary (optional if config_path provided)

    Returns:
        Tuple of (images, l0_annotations, l1_annotations)
    """
    from .core import load_config

    if config is None:
        if config_path is None:
            config_path = 'configs/hierarchical-yolo-seg.yaml'
        config = load_config(config_path)

    if not is_hierarchical_mode(config):
        print("ERROR: Hierarchical mode not enabled in config")
        print("Set hierarchy.enabled: true in your config file")
        return 0, 0, 0

    base_output_dir = config.get('output_base_dir', 'yolo_hierarchical')
    os.makedirs(base_output_dir, exist_ok=True)

    # Save config snapshot
    import yaml
    config_snapshot = os.path.join(base_output_dir, 'config_used.yaml')
    with open(config_snapshot, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("=" * 70)
    print("HIERARCHICAL DATASET GENERATION (H-DETR)")
    print("=" * 70)
    print(f"Output: {base_output_dir}")
    print(f"L0: County ({config['hierarchy']['level_0']['num_classes']} classes)")
    print(f"L1: Township ({config['hierarchy']['level_1']['num_classes']} classes)")
    print("=" * 70)

    # Run processing - accumulate totals from both modes
    total_images = 0
    total_l0 = 0
    total_l1 = 0

    # Process separate districts mode
    if config.get('processing_modes', {}).get('separate_districts', True):
        imgs, l0, l1 = process_hierarchical_separate_districts(config, base_output_dir)
        total_images += imgs
        total_l0 += l0
        total_l1 += l1

    # Process combined maps mode
    if config.get('processing_modes', {}).get('combined_maps', False):
        imgs, l0, l1 = process_hierarchical_combined_maps(config, base_output_dir)
        total_images += imgs
        total_l0 += l0
        total_l1 += l1

    print("\n" + "=" * 70)
    print("HIERARCHICAL GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total Images: {total_images}")
    print(f"Total L0 (County) annotations: {total_l0}")
    print(f"Total L1 (Township) annotations: {total_l1}")
    print("=" * 70)

    return total_images, total_l0, total_l1
