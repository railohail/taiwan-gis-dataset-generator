"""
Hierarchical YOLO format annotation writer for H-DETR.

This module handles:
- Writing dual-level YOLO annotations (L0: county, L1: township)
- Each image gets TWO label files: image_l0.txt and image_l1.txt
- Dataset structure with classes_l0.txt and classes_l1.txt
- Hierarchical dataset.yaml for training

Output structure:
    yolo_hierarchical/
    ├── classes_l0.txt        # 22 county classes
    ├── classes_l1.txt        # 368 township classes
    ├── hierarchy.json        # Copy of admin hierarchy
    ├── dataset.yaml          # Combined dataset config
    ├── train/
    │   ├── images/
    │   │   └── map_001.png
    │   └── labels/
    │       ├── map_001_l0.txt    # County annotations
    │       └── map_001_l1.txt    # Township annotations
    ├── val/
    └── test/
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .yolo import coco_segmentation_to_yolo, coco_to_yolo_bbox


def write_hierarchical_yolo_annotation_files(
    image_name: str,
    annotations_l0: List[Dict],
    annotations_l1: List[Dict],
    image_width: int,
    image_height: int,
    output_dir: str,
    annotation_type: str = 'segmentation'
) -> Tuple[str, str]:
    """
    Write hierarchical YOLO annotation files for a single image.

    Creates TWO files:
    - {image_name}_l0.txt - County level annotations
    - {image_name}_l1.txt - Township level annotations

    Args:
        image_name: Name of the image file (e.g., "map_001.png")
        annotations_l0: List of COCO annotations for county level
        annotations_l1: List of COCO annotations for township level
        image_width: Image width in pixels
        image_height: Image height in pixels
        output_dir: Directory to save annotation files
        annotation_type: 'bbox' or 'segmentation'

    Returns:
        Tuple of (l0_path, l1_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(image_name)[0]

    # Write L0 (county) annotations
    l0_path = os.path.join(output_dir, f"{base_name}_l0.txt")
    _write_yolo_file(l0_path, annotations_l0, image_width, image_height, annotation_type)

    # Write L1 (township) annotations
    l1_path = os.path.join(output_dir, f"{base_name}_l1.txt")
    _write_yolo_file(l1_path, annotations_l1, image_width, image_height, annotation_type)

    return l0_path, l1_path


def _write_yolo_file(
    path: str,
    annotations: List[Dict],
    image_width: int,
    image_height: int,
    annotation_type: str
):
    """Write a single YOLO annotation file."""
    with open(path, 'w', encoding='utf-8') as f:
        for ann in annotations:
            class_id = ann['category_id']

            if annotation_type == 'segmentation':
                if 'segmentation' in ann and ann['segmentation']:
                    yolo_seg = coco_segmentation_to_yolo(
                        ann['segmentation'], image_width, image_height
                    )
                    if yolo_seg and len(yolo_seg) >= 6:
                        coords_str = ' '.join([f"{coord:.6f}" for coord in yolo_seg])
                        f.write(f"{class_id} {coords_str}\n")
            else:
                coco_bbox = ann['bbox']
                yolo_bbox = coco_to_yolo_bbox(coco_bbox, image_width, image_height)
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                       f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")


def initialize_hierarchical_yolo_dataset(
    base_output_dir: str,
    categories_l0: List[Dict],
    categories_l1: List[Dict],
    hierarchy_file: Optional[str] = None,
    use_split: bool = True
) -> Dict[str, str]:
    """
    Initialize hierarchical YOLO dataset structure.

    Creates:
    - classes_l0.txt (county classes)
    - classes_l1.txt (township classes)
    - dataset.yaml (hierarchical config)
    - hierarchy.json (copy of admin hierarchy)
    - train/val/test directories

    Args:
        base_output_dir: Base output directory
        categories_l0: List of county categories
        categories_l1: List of township categories
        hierarchy_file: Path to hierarchy JSON to copy
        use_split: Whether to create train/val/test splits

    Returns:
        Dict with paths to created files
    """
    os.makedirs(base_output_dir, exist_ok=True)

    paths = {}

    # Create split directories
    if use_split:
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(base_output_dir, split, 'images')
            labels_dir = os.path.join(base_output_dir, split, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
    else:
        os.makedirs(os.path.join(base_output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, 'labels'), exist_ok=True)

    # Write classes_l0.txt (counties)
    paths['classes_l0'] = os.path.join(base_output_dir, 'classes_l0.txt')
    sorted_l0 = sorted(categories_l0, key=lambda x: x['id'])
    with open(paths['classes_l0'], 'w', encoding='utf-8') as f:
        for cat in sorted_l0:
            f.write(f"{cat['name']}\n")
    print(f"Created classes_l0.txt: {len(categories_l0)} county classes")

    # Write classes_l1.txt (townships)
    paths['classes_l1'] = os.path.join(base_output_dir, 'classes_l1.txt')
    sorted_l1 = sorted(categories_l1, key=lambda x: x['id'])
    with open(paths['classes_l1'], 'w', encoding='utf-8') as f:
        for cat in sorted_l1:
            f.write(f"{cat['name']}\n")
    print(f"Created classes_l1.txt: {len(categories_l1)} township classes")

    # Copy hierarchy.json
    if hierarchy_file and os.path.exists(hierarchy_file):
        paths['hierarchy'] = os.path.join(base_output_dir, 'hierarchy.json')
        shutil.copy(hierarchy_file, paths['hierarchy'])
        print(f"Copied hierarchy.json")

    # Write hierarchical dataset.yaml
    paths['dataset_yaml'] = os.path.join(base_output_dir, 'dataset.yaml')
    _write_hierarchical_dataset_yaml(
        paths['dataset_yaml'],
        categories_l0,
        categories_l1,
        use_split
    )
    print(f"Created hierarchical dataset.yaml")

    return paths


def _write_hierarchical_dataset_yaml(
    path: str,
    categories_l0: List[Dict],
    categories_l1: List[Dict],
    use_split: bool
):
    """Write hierarchical dataset.yaml configuration."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Hierarchical YOLO Dataset Configuration (H-DETR)\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Two-level hierarchy: County (L0) -> Township (L1)\n\n")

        f.write("# Dataset paths\n")
        f.write("path: .  # dataset root dir\n")

        if use_split:
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("test: test/images\n\n")
        else:
            f.write("train: images\n")
            f.write("val: images\n\n")

        f.write("# Label file naming convention:\n")
        f.write("# For image 'map_001.png', labels are:\n")
        f.write("#   - labels/map_001_l0.txt (county level)\n")
        f.write("#   - labels/map_001_l1.txt (township level)\n\n")

        f.write("# Hierarchical structure\n")
        f.write("hierarchical: true\n")
        f.write("hierarchy_file: hierarchy.json\n\n")

        # Level 0 (County)
        f.write("# Level 0: Counties (縣市)\n")
        f.write(f"nc_l0: {len(categories_l0)}\n")
        f.write("names_l0:\n")
        sorted_l0 = sorted(categories_l0, key=lambda x: x['id'])
        for cat in sorted_l0:
            f.write(f"  {cat['id']}: {cat['name']}\n")

        f.write("\n")

        # Level 1 (Township)
        f.write("# Level 1: Townships (鄉鎮區)\n")
        f.write(f"nc_l1: {len(categories_l1)}\n")
        f.write("names_l1:\n")
        sorted_l1 = sorted(categories_l1, key=lambda x: x['id'])
        for cat in sorted_l1:
            parent_info = f"  # parent: {cat.get('parent_id', '?')}" if 'parent_id' in cat else ""
            f.write(f"  {cat['id']}: {cat['name']}{parent_info}\n")


# Batch buffer for hierarchical YOLO
# IMPORTANT: Each entry now tracks its target labels_dir to support multiple splits
_hierarchical_batch_buffer = {
    'entries': [],  # List of (image_info, annotations_l0, annotations_l1, labels_dir)
}
_hierarchical_batch_size = 100


def set_hierarchical_batch_size(size: int):
    """Set batch size for hierarchical writes."""
    global _hierarchical_batch_size
    _hierarchical_batch_size = size


def batch_append_hierarchical(
    image_info: Dict,
    annotations_l0: List[Dict],
    annotations_l1: List[Dict],
    labels_dir: str = None
):
    """
    Add image and dual-level annotations to batch buffer.

    Args:
        image_info: Image info dict with id, width, height, file_name
        annotations_l0: County level annotations
        annotations_l1: Township level annotations
        labels_dir: Target directory for this image's labels (REQUIRED for correct split handling)
    """
    global _hierarchical_batch_buffer

    _hierarchical_batch_buffer['entries'].append({
        'image_info': image_info,
        'annotations_l0': annotations_l0,
        'annotations_l1': annotations_l1,
        'labels_dir': labels_dir
    })


def flush_hierarchical_batch(
    labels_dir: str = None,
    annotation_type: str = 'segmentation',
    force: bool = False
) -> int:
    """
    Flush hierarchical batch buffer to disk.

    Each entry is written to its own stored labels_dir, ensuring correct
    split handling when processing train/val/test in sequence.

    Args:
        labels_dir: Fallback directory if entry doesn't have one stored
        annotation_type: 'bbox' or 'segmentation'
        force: If True, flush regardless of buffer size

    Returns:
        Number of images written
    """
    global _hierarchical_batch_buffer

    if not _hierarchical_batch_buffer['entries'] and not force:
        return 0

    if len(_hierarchical_batch_buffer['entries']) < _hierarchical_batch_size and not force:
        return 0

    count = 0
    total_l0 = 0
    total_l1 = 0

    for entry in _hierarchical_batch_buffer['entries']:
        image_info = entry['image_info']
        annotations_l0 = entry['annotations_l0']
        annotations_l1 = entry['annotations_l1']
        # Use entry's stored labels_dir, fall back to passed labels_dir
        target_dir = entry['labels_dir'] or labels_dir

        if not target_dir:
            print(f"WARNING: No labels_dir for image {image_info['file_name']}, skipping")
            continue

        image_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        write_hierarchical_yolo_annotation_files(
            image_name,
            annotations_l0,
            annotations_l1,
            image_width,
            image_height,
            target_dir,
            annotation_type
        )
        count += 1
        total_l0 += len(annotations_l0)
        total_l1 += len(annotations_l1)

    if count > 0:
        print(f"    HIERARCHICAL BATCH FLUSH: Wrote {count} images "
              f"(L0: {total_l0} anns, L1: {total_l1} anns)")

    # Clear buffer
    _hierarchical_batch_buffer['entries'].clear()

    return count


def reset_hierarchical_batch():
    """Reset the hierarchical batch buffer."""
    global _hierarchical_batch_buffer
    _hierarchical_batch_buffer['entries'].clear()
