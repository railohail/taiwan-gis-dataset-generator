"""
Annotation utilities for GIS dataset generation.

This module handles:
- COCO format annotation creation and serialization
- Polygon to segmentation conversion
- Bounding box and area calculations
- Batch annotation buffering for performance

Supports both COCO JSON and YOLO txt output formats.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon as ShapelyPolygon

from .config import COUNTY_TO_CLASS
from .geometry import clip_polygon_to_window
from .core.constants import MIN_POLYGON_AREA, MIN_SEGMENTATION_LENGTH


def polygon_to_coco_segmentation(
    polygon: ShapelyPolygon,
    image_height: int,
    image_width: int
) -> List[float]:
    """
    Convert a Shapely polygon to COCO segmentation format.

    Args:
        polygon: Shapely Polygon object
        image_height: Image height in pixels
        image_width: Image width in pixels

    Returns:
        List of [x1, y1, x2, y2, ...] coordinates
    """
    if polygon is None or polygon.is_empty:
        return []

    pixel_coords = []
    for x, y in polygon.exterior.coords[:-1]:
        px = max(0, min(x, image_width - 1))
        py = max(0, min(y, image_height - 1))
        pixel_coords.extend([px, py])

    return pixel_coords


def calculate_bbox_from_segmentation(segmentation: List[float]) -> List[float]:
    """
    Calculate bounding box from segmentation coordinates.

    Args:
        segmentation: List of [x1, y1, x2, y2, ...] coordinates

    Returns:
        Bounding box as [x, y, width, height]
    """
    if not segmentation:
        return [0, 0, 0, 0]

    x_coords = segmentation[0::2]
    y_coords = segmentation[1::2]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def calculate_area_from_segmentation(segmentation: List[float]) -> float:
    """Calculate area of polygon from segmentation coordinates using the shoelace formula."""
    if len(segmentation) < MIN_SEGMENTATION_LENGTH:
        return 0.0

    points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]

    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2.0


def get_county_name_from_row(row, columns):
    """Extract county name from shapefile row."""
    county_name = None
    possible_columns = ['COUNTYNAME', 'NAME', 'County', 'county', '縣市名稱', 'COUNTY', 'COUNTYENG', 'C_Name']
    for col in possible_columns:
        if col in columns and row[col] is not None:
            county_name = str(row[col]).strip()
            if county_name and county_name != 'None':
                break
    return county_name


def match_county_category(county_name):
    """Match county name to category ID with variation handling."""
    category_id = COUNTY_TO_CLASS.get(county_name)
    if category_id is None:
        # Try common variations
        variations = [
            county_name.replace('台', '臺'),  # Traditional vs Simplified
            county_name.replace('臺', '台'),  # Traditional vs Simplified
        ]
        for variant in variations:
            category_id = COUNTY_TO_CLASS.get(variant)
            if category_id is not None:
                break
    return category_id


def create_annotations_for_image(clipped_shapefile, image_shape, transform, image_id):
    """Create COCO annotations for an image."""
    annotations = []
    annotation_id = 1

    for idx, row in clipped_shapefile.iterrows():
        geometry = row.geometry

        # Get county name
        county_name = get_county_name_from_row(row, clipped_shapefile.columns)
        if not county_name:
            continue

        # Match category ID
        category_id = match_county_category(county_name)
        if category_id is None:
            continue

        # Handle MultiPolygon case - process ALL parts, not just the largest
        geometries_to_process = []
        if geometry.geom_type == 'MultiPolygon':
            # Keep all parts that are large enough
            geometries_to_process = [g for g in geometry.geoms if g.area > 0 and g.geom_type == 'Polygon']
        elif geometry.geom_type == 'Polygon':
            geometries_to_process = [geometry]
        else:
            continue

        # Process each polygon part
        for geom in geometries_to_process:
            # Convert geometry to pixel coordinates
            pixel_coords = []
            for x, y in geom.exterior.coords:
                px, py = ~transform * (x, y)
                pixel_coords.append([px, py])

            if len(pixel_coords) < 3:
                continue

            # Convert to segmentation format
            segmentation = []
            for px, py in pixel_coords[:-1]:  # Exclude last duplicate
                segmentation.extend([max(0, min(px, image_shape[1]-1)),
                                   max(0, min(py, image_shape[0]-1))])

            if len(segmentation) < MIN_SEGMENTATION_LENGTH:
                continue

            bbox = calculate_bbox_from_segmentation(segmentation)
            area = calculate_area_from_segmentation(segmentation)

            # Skip if area too small
            if area < MIN_POLYGON_AREA:
                continue

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }

            annotations.append(annotation)
            annotation_id += 1

    return annotations


def create_annotations_for_window(clipped_shapefile, window_info, cropped_transform):
    """Create COCO annotations for a specific window."""
    annotations = []
    annotation_id = 1

    for idx, row in clipped_shapefile.iterrows():
        geometry = row.geometry

        # Get county name
        county_name = get_county_name_from_row(row, clipped_shapefile.columns)
        if not county_name:
            continue

        # Match category ID
        category_id = match_county_category(county_name)
        if category_id is None:
            continue

        # Handle MultiPolygon case - process ALL parts, not just the largest
        geometries_to_process = []
        if geometry.geom_type == 'MultiPolygon':
            geometries_to_process = [g for g in geometry.geoms if g.area > 0 and g.geom_type == 'Polygon']
        elif geometry.geom_type == 'Polygon':
            geometries_to_process = [geometry]
        else:
            continue

        # Process each polygon part
        for geom in geometries_to_process:
            # Convert geometry to pixel coordinates
            pixel_coords = []
            for x, y in geom.exterior.coords:
                px, py = ~cropped_transform * (x, y)
                pixel_coords.append([px, py])

            if len(pixel_coords) < 3:
                continue

            pixel_polygon = ShapelyPolygon(pixel_coords)

            # Clip polygon to window (may return Polygon or MultiPolygon)
            clipped_polygon = clip_polygon_to_window(
                pixel_polygon,
                window_info['start_x'],
                window_info['start_y'],
                window_info['end_x'],
                window_info['end_y']
            )

            if clipped_polygon is None:
                continue

            # Handle both Polygon and MultiPolygon results from clipping
            polygons_to_annotate = []
            if clipped_polygon.geom_type == 'MultiPolygon':
                # Keep all parts that are large enough
                polygons_to_annotate = [p for p in clipped_polygon.geoms if p.area >= 10]
            elif clipped_polygon.geom_type == 'Polygon':
                if clipped_polygon.area >= 10:
                    polygons_to_annotate = [clipped_polygon]

            # Create annotation for each polygon part
            for poly in polygons_to_annotate:
                # Convert to COCO segmentation format
                segmentation = polygon_to_coco_segmentation(
                    poly,
                    window_info['height'],
                    window_info['width']
                )

                if len(segmentation) < MIN_SEGMENTATION_LENGTH:
                    continue

                bbox = calculate_bbox_from_segmentation(segmentation)
                area = calculate_area_from_segmentation(segmentation)

                if area < MIN_POLYGON_AREA:
                    continue

                annotation = {
                    "id": annotation_id,
                    "image_id": window_info['id'] + 1,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }

                annotations.append(annotation)
                annotation_id += 1

    return annotations


def initialize_coco_file(base_output_dir, use_split=False):
    """
    Initialize the COCO JSON file with basic structure.

    Args:
        base_output_dir: Base output directory
        use_split: If True, create train/val/test split structure

    Returns:
        Path to the main COCO file (or train file if using splits)
    """
    from .config import create_categories

    if use_split:
        # Create Coconuts-1 style structure: train/val/test folders with images and _annotations.coco.json in each
        train_dir = os.path.join(base_output_dir, 'train')
        val_dir = os.path.join(base_output_dir, 'valid')  # Use 'valid' like Coconuts-1
        test_dir = os.path.join(base_output_dir, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Initialize separate COCO files for each split (Coconuts-1 style: _annotations.coco.json)
        categories = create_categories()
        splits = [
            ('train', train_dir),
            ('valid', val_dir),  # Use 'valid' name for consistency
            ('test', test_dir)
        ]

        for split_name, split_dir in splits:
            coco_file_path = os.path.join(split_dir, '_annotations.coco.json')

            initial_coco_data = {
                "info": {
                    "description": f"Taiwan Counties Dataset - {split_name.upper()} split",
                    "url": "",
                    "version": "2.0",
                    "year": 2024,
                    "contributor": "Full COCO Generator v2",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [{"url": "", "id": 1, "name": "Unknown"}],
                "images": [],
                "annotations": [],
                "categories": categories
            }

            with open(coco_file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_coco_data, f, indent=2, ensure_ascii=False)

            print(f"Initialized COCO file: {coco_file_path}")

        # Return the train file path as the main one
        return os.path.join(train_dir, '_annotations.coco.json')
    else:
        # Original single-file behavior
        annotations_dir = os.path.join(base_output_dir, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        coco_file_path = os.path.join(annotations_dir, 'annotations.json')

        categories = create_categories()

        initial_coco_data = {
            "info": {
                "description": "Taiwan Counties Dataset - v2 (Separate + Combined + Hue Augmentation)",
                "url": "",
                "version": "2.0",
                "year": 2024,
                "contributor": "Full COCO Generator v2",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"url": "", "id": 1, "name": "Unknown"}],
            "images": [],
            "annotations": [],
            "categories": categories
        }

        with open(coco_file_path, 'w', encoding='utf-8') as f:
            json.dump(initial_coco_data, f, indent=2, ensure_ascii=False)

        print(f"Initialized COCO file: {coco_file_path}")
        return coco_file_path


def verify_coco_consistency(coco_file_path, new_images, new_annotations):
    """Verify that new images and annotations are consistent."""
    issues = []

    # Create image ID to image info mapping
    image_map = {img['id']: img for img in new_images}

    for ann in new_annotations:
        image_id = ann['image_id']
        if image_id not in image_map:
            issues.append(f"Annotation {ann['id']} references non-existent image {image_id}")
            continue

        image_info = image_map[image_id]
        image_width, image_height = image_info['width'], image_info['height']

        # Check segmentation bounds
        segmentation = ann['segmentation'][0]
        for i in range(0, len(segmentation), 2):
            x, y = segmentation[i], segmentation[i+1]
            if x < 0 or x >= image_width or y < 0 or y >= image_height:
                issues.append(f"Annotation {ann['id']}: point ({x:.1f}, {y:.1f}) outside image {image_id} bounds ({image_width}x{image_height})")

        # Check bbox bounds
        x, y, w, h = ann['bbox']
        if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
            issues.append(f"Annotation {ann['id']}: bbox ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}) outside image {image_id} bounds")

    if issues:
        print(f"    COCO consistency issues found:")
        for issue in issues[:3]:
            print(f"      - {issue}")
        if len(issues) > 3:
            print(f"      - ... and {len(issues) - 3} more issues")
        return False

    print(f"    COCO consistency verified: {len(new_images)} images, {len(new_annotations)} annotations")
    return True


def convert_annotations_to_bbox_only(annotations):
    """
    Convert annotations to bbox-only format by removing segmentation data.

    Args:
        annotations: List of COCO annotations

    Returns:
        List of bbox-only annotations
    """
    bbox_annotations = []
    for ann in annotations:
        bbox_ann = {
            "id": ann['id'],
            "image_id": ann['image_id'],
            "category_id": ann['category_id'],
            "bbox": ann['bbox'],
            "area": ann['area'],
            "iscrowd": ann['iscrowd']
        }
        # Explicitly omit 'segmentation' field
        bbox_annotations.append(bbox_ann)

    return bbox_annotations


def append_to_coco_file(coco_file_path, new_images, new_annotations):
    """Append new images and annotations to the existing COCO file.

    OPTIMIZATION: This function has been replaced with batch processing.
    Use batch_append_to_coco_file() instead for better performance.
    This function is kept for backward compatibility but should be avoided.
    """
    try:
        # Verify consistency before writing
        if not verify_coco_consistency(coco_file_path, new_images, new_annotations):
            print("    WARNING: COCO consistency issues detected but proceeding...")

        # Read current data
        with open(coco_file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Append new data
        coco_data['images'].extend(new_images)
        coco_data['annotations'].extend(new_annotations)

        # Write back
        with open(coco_file_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"    COCO file updated: +{len(new_images)} images, +{len(new_annotations)} annotations")

    except Exception as e:
        print(f"    ERROR in append_to_coco_file: {e}")
        raise


# OPTIMIZATION: Batch buffer for COCO file writes
# Use dict to maintain separate buffers per split/file to prevent cross-contamination
_coco_batch_buffers = {}  # Key: coco_file_path, Value: {'images': [], 'annotations': []}
_coco_batch_size = 100  # Default batch size


def set_batch_size(size):
    """Set the batch size for buffered writes."""
    global _coco_batch_size
    _coco_batch_size = size


def _get_buffer_for_file(coco_file_path):
    """Get or create buffer for specific COCO file."""
    global _coco_batch_buffers
    if coco_file_path not in _coco_batch_buffers:
        _coco_batch_buffers[coco_file_path] = {'images': [], 'annotations': []}
    return _coco_batch_buffers[coco_file_path]


def batch_append_to_coco_buffer(new_images, new_annotations, annotation_type='segmentation', coco_file_path=None):
    """
    Add images and annotations to batch buffer for a specific COCO file.

    Args:
        new_images: List of image info dicts
        new_annotations: List of annotation dicts
        annotation_type: 'segmentation' or 'bbox'
        coco_file_path: Path to COCO file (required for split mode)
    """
    # For backward compatibility, if no path specified, use default buffer
    if coco_file_path is None:
        # Legacy mode: use global buffer key
        coco_file_path = '__default__'

    buffer = _get_buffer_for_file(coco_file_path)
    buffer['images'].extend(new_images)

    # Convert to bbox-only if requested
    if annotation_type == 'bbox':
        new_annotations = convert_annotations_to_bbox_only(new_annotations)

    buffer['annotations'].extend(new_annotations)


def flush_coco_batch(coco_file_path, force=False):
    """Flush batch buffer for specific COCO file."""
    # For backward compatibility
    if coco_file_path is None:
        coco_file_path = '__default__'

    buffer = _get_buffer_for_file(coco_file_path)

    if not buffer['images'] and not force:
        return 0

    if len(buffer['images']) < _coco_batch_size and not force:
        return 0

    try:
        # Read current data
        with open(coco_file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Append buffered data
        images_count = len(buffer['images'])
        annotations_count = len(buffer['annotations'])

        coco_data['images'].extend(buffer['images'])
        coco_data['annotations'].extend(buffer['annotations'])

        # Write back
        with open(coco_file_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"    BATCH FLUSH: Wrote {images_count} images, {annotations_count} annotations to {os.path.basename(coco_file_path)}")

        # Clear buffer for this file
        buffer['images'].clear()
        buffer['annotations'].clear()

        return images_count

    except Exception as e:
        print(f"    ERROR in flush_coco_batch: {e}")
        raise


def flush_all_coco_batches(force=True):
    """Flush all batch buffers to their respective COCO files."""
    global _coco_batch_buffers
    total_flushed = 0
    for coco_file_path in list(_coco_batch_buffers.keys()):
        if coco_file_path != '__default__':  # Skip legacy default buffer
            flushed = flush_coco_batch(coco_file_path, force=force)
            total_flushed += flushed
    return total_flushed


def reset_coco_batch():
    """Reset all batch buffers (useful between different processing modes)."""
    global _coco_batch_buffers
    for buffer in _coco_batch_buffers.values():
        buffer['images'].clear()
        buffer['annotations'].clear()
    _coco_batch_buffers.clear()


# =============================================================================
# FAST COCO MODE: Temp-File System with Atomic Writes
# =============================================================================
# This system writes individual JSON files per image during processing (YOLO-speed),
# then merges them into final COCO format at the end (safe, atomic).
#
# Architecture:
#   During processing: output_dir/temp_annotations/{image_id:07d}.json
#   After processing:  output_dir/[split/]_annotations.coco.json
#
# Benefits:
#   - Same speed as YOLO during processing (independent file writes)
#   - Atomic merge prevents corruption on Ctrl+C
#   - Standard COCO output format
#   - Crash recovery (temp files preserved)
# =============================================================================

import shutil
from pathlib import Path
from tqdm import tqdm


# Global temp-file mode state
_temp_file_mode = {
    'enabled': False,
    'temp_dir': None,
    'split_temp_dirs': {}  # Key: split name, Value: temp dir path
}


def enable_temp_file_mode(base_output_dir, use_split=False):
    """
    Enable temp-file mode for fast COCO annotation writing.

    Args:
        base_output_dir: Base output directory
        use_split: If True, create separate temp dirs for train/val/test

    Returns:
        Path to temp directory (or dict of split -> temp_dir if use_split=True)
    """
    global _temp_file_mode

    if use_split:
        # Create temp dirs for each split
        splits = ['train', 'valid', 'test']
        temp_dirs = {}
        for split in splits:
            temp_dir = os.path.join(base_output_dir, split, 'temp_annotations')
            os.makedirs(temp_dir, exist_ok=True)
            temp_dirs[split] = temp_dir

        _temp_file_mode['enabled'] = True
        _temp_file_mode['temp_dir'] = None
        _temp_file_mode['split_temp_dirs'] = temp_dirs

        print(f"[OK] Temp-file mode enabled (split mode) - directories created")
        return temp_dirs
    else:
        # Single temp dir
        temp_dir = os.path.join(base_output_dir, 'temp_annotations')
        os.makedirs(temp_dir, exist_ok=True)

        _temp_file_mode['enabled'] = True
        _temp_file_mode['temp_dir'] = temp_dir
        _temp_file_mode['split_temp_dirs'] = {}

        print(f"[OK] Temp-file mode enabled - directory: {temp_dir}")
        return temp_dir


def is_temp_file_mode_enabled():
    """Check if temp-file mode is enabled."""
    return _temp_file_mode['enabled']


def write_temp_annotation_file(image_info, annotations, temp_dir, annotation_type='segmentation'):
    """
    Write a single temp JSON file for one image (YOLO-style speed).

    Args:
        image_info: Image info dict
        annotations: List of annotations for this image
        temp_dir: Directory to write temp file
        annotation_type: 'segmentation' or 'bbox'

    Returns:
        Path to temp file
    """
    image_id = image_info['id']

    # Convert to bbox-only if requested
    if annotation_type == 'bbox':
        annotations = convert_annotations_to_bbox_only(annotations)

    # Write temp JSON file
    temp_file_path = os.path.join(temp_dir, f"{image_id:07d}.json")

    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'image': image_info,
            'annotations': annotations
        }, f, ensure_ascii=False)

    return temp_file_path


def merge_temp_coco_files(temp_dir, final_coco_path, categories, annotation_type='segmentation'):
    """
    Merge all temp JSON files into final COCO format with ATOMIC write.

    This function is SAFE against Ctrl+C:
    - Builds COCO structure in memory
    - Writes to .tmp file first
    - Atomically renames to final path
    - Only deletes temp files after successful write

    Args:
        temp_dir: Directory containing temp JSON files
        final_coco_path: Path to final COCO JSON file
        categories: List of category dicts
        annotation_type: 'segmentation' or 'bbox' (for verification)

    Returns:
        Tuple of (images_count, annotations_count)
    """
    if not os.path.exists(temp_dir):
        print(f"⚠️  No temp directory found: {temp_dir}")
        return 0, 0

    # Find all temp files
    temp_files = sorted(Path(temp_dir).glob("*.json"))

    if not temp_files:
        print(f"⚠️  No temp annotation files found in {temp_dir}")
        return 0, 0

    print(f"\n{'='*60}")
    print(f"Merging {len(temp_files)} temp annotation files...")
    print(f"{'='*60}")

    # Build COCO structure in memory
    coco_data = {
        "info": {
            "description": "Taiwan County GIS Dataset (COCO Format)",
            "version": "2.0",
            "year": datetime.now().year,
            "contributor": "GIS Dataset Generator",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    # Merge all temp files with progress bar
    for temp_file in tqdm(temp_files, desc="Merging", unit="files"):
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                coco_data['images'].append(data['image'])
                coco_data['annotations'].extend(data['annotations'])
        except Exception as e:
            print(f"\n⚠️  Error reading {temp_file}: {e}")
            continue

    images_count = len(coco_data['images'])
    annotations_count = len(coco_data['annotations'])

    print(f"[OK] Merged: {images_count} images, {annotations_count} annotations")

    # ATOMIC WRITE: Write to temp location first
    os.makedirs(os.path.dirname(final_coco_path), exist_ok=True)
    temp_output_path = final_coco_path + '.tmp'

    print(f"Writing to temporary file: {os.path.basename(temp_output_path)}...")
    with open(temp_output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    # Atomic rename (safe operation on most OS)
    print(f"Atomic rename to: {os.path.basename(final_coco_path)}...")
    os.replace(temp_output_path, final_coco_path)

    print(f"[OK] Final COCO file created: {final_coco_path}")

    # Only delete temp files AFTER successful merge
    print(f"Cleaning up {len(temp_files)} temp files...")
    try:
        shutil.rmtree(temp_dir)
        print(f"[OK] Temp directory deleted: {temp_dir}")
    except Exception as e:
        print(f"[WARNING] Could not delete temp directory: {e}")
        print(f"   You can manually delete: {temp_dir}")

    print(f"{'='*60}\n")

    return images_count, annotations_count


def merge_all_split_temp_files(base_output_dir, categories, annotation_type='segmentation'):
    """
    Merge temp files for all splits (train/valid/test).

    Args:
        base_output_dir: Base output directory
        categories: List of category dicts
        annotation_type: 'segmentation' or 'bbox'

    Returns:
        Dict of split -> (images_count, annotations_count)
    """
    global _temp_file_mode

    results = {}
    splits = ['train', 'valid', 'test']

    for split in splits:
        temp_dir = os.path.join(base_output_dir, split, 'temp_annotations')
        final_coco_path = os.path.join(base_output_dir, split, '_annotations.coco.json')

        if os.path.exists(temp_dir):
            print(f"\n{'='*60}")
            print(f"Processing {split.upper()} split...")
            print(f"{'='*60}")

            images_count, annotations_count = merge_temp_coco_files(
                temp_dir, final_coco_path, categories, annotation_type
            )
            results[split] = (images_count, annotations_count)

    return results


def disable_temp_file_mode():
    """Disable temp-file mode and reset state."""
    global _temp_file_mode
    _temp_file_mode = {
        'enabled': False,
        'temp_dir': None,
        'split_temp_dirs': {}
    }
