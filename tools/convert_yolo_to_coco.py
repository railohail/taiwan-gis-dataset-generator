"""
Fast YOLO Segmentation to COCO Format Converter

Converts YOLO segmentation dataset to COCO format matching the exact structure
produced by main.py.

Usage:
    python -m tools.convert_yolo_to_coco
    python -m tools.convert_yolo_to_coco --input yolo_seg_base --output coco_seg_base

Features:
    - Fast conversion (processes images in parallel)
    - Exact COCO format matching main.py output
    - Preserves train/val/test splits
    - Handles YOLO segmentation polygons
    - Coconuts-1 style split structure
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import argparse


def load_class_names(yolo_dir: str) -> list:
    """Load class names from classes.txt."""
    classes_path = os.path.join(yolo_dir, 'classes.txt')
    with open(classes_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def create_categories(class_names: list) -> list:
    """Create COCO categories from class names."""
    categories = []
    for idx, name in enumerate(class_names):
        categories.append({
            'id': idx,
            'name': name,
            'supercategory': 'region'
        })
    return categories


def yolo_segmentation_to_coco(yolo_coords: list, img_width: int, img_height: int) -> list:
    """
    Convert YOLO normalized segmentation to COCO pixel coordinates.

    Args:
        yolo_coords: List of normalized coordinates [x1, y1, x2, y2, ...]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of pixel coordinates [px1, py1, px2, py2, ...]
    """
    coco_seg = []
    for i in range(0, len(yolo_coords), 2):
        x_norm = yolo_coords[i]
        y_norm = yolo_coords[i + 1]
        px = x_norm * img_width
        py = y_norm * img_height
        coco_seg.extend([px, py])
    return coco_seg


def calculate_bbox_from_segmentation(segmentation: list) -> list:
    """Calculate bounding box from segmentation coordinates."""
    if not segmentation:
        return [0, 0, 0, 0]

    x_coords = segmentation[0::2]
    y_coords = segmentation[1::2]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def calculate_area_from_segmentation(segmentation: list) -> float:
    """Calculate area of polygon from segmentation coordinates (Shoelace formula)."""
    if len(segmentation) < 6:
        return 0.0

    points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]

    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2.0


def parse_yolo_label(label_path: str, img_width: int, img_height: int) -> list:
    """
    Parse YOLO label file and convert to COCO annotations.

    Args:
        label_path: Path to YOLO .txt label file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of annotation dicts (without image_id and id)
    """
    annotations = []

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + at least 3 points (x,y)
            continue

        class_id = int(parts[0])

        # Parse normalized coordinates
        yolo_coords = [float(x) for x in parts[1:]]

        # Convert to COCO pixel coordinates
        coco_seg = yolo_segmentation_to_coco(yolo_coords, img_width, img_height)

        if len(coco_seg) < 6:  # Need at least 3 points
            continue

        # Calculate bbox and area
        bbox = calculate_bbox_from_segmentation(coco_seg)
        area = calculate_area_from_segmentation(coco_seg)

        if area < 10:  # Skip tiny polygons
            continue

        annotation = {
            'category_id': class_id,
            'segmentation': [coco_seg],
            'bbox': bbox,
            'area': area,
            'iscrowd': 0
        }

        annotations.append(annotation)

    return annotations


def convert_split(yolo_split_dir: str, coco_split_dir: str, split_name: str, categories: list) -> tuple:
    """
    Convert one split (train/val/test) from YOLO to COCO format.

    Args:
        yolo_split_dir: Path to YOLO split directory (e.g., yolo_seg_base/train)
        coco_split_dir: Path to COCO split directory (e.g., coco_seg_base/train)
        split_name: Name of split ('train', 'valid', 'test')
        categories: List of COCO category dicts

    Returns:
        Tuple of (images_count, annotations_count)
    """
    images_dir = os.path.join(yolo_split_dir, 'images')
    labels_dir = os.path.join(yolo_split_dir, 'labels')

    # Create output directories
    coco_images_dir = coco_split_dir
    os.makedirs(coco_images_dir, exist_ok=True)

    # Copy visualizations if they exist
    vis_src = os.path.join(yolo_split_dir, 'visualizations')
    if os.path.exists(vis_src):
        vis_dst = os.path.join(os.path.dirname(coco_split_dir), 'visualizations', split_name)
        os.makedirs(vis_dst, exist_ok=True)
        print(f"  Copying visualizations to {vis_dst}...")
        for vis_file in os.listdir(vis_src):
            shutil.copy2(os.path.join(vis_src, vis_file), os.path.join(vis_dst, vis_file))

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": f"Taiwan Counties Dataset - {split_name.upper()} split (Converted from YOLO)",
            "url": "",
            "version": "2.0",
            "year": datetime.now().year,
            "contributor": "YOLO to COCO Converter",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{"url": "", "id": 1, "name": "Unknown"}],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    print(f"  Processing {len(image_files)} images...")

    image_id = 0
    annotation_id = 0

    for img_file in tqdm(image_files, desc=f"  {split_name}", unit="img"):
        # Copy image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(coco_images_dir, img_file)
        shutil.copy2(src_img, dst_img)

        # Get image dimensions
        with Image.open(src_img) as img:
            img_width, img_height = img.size

        # Add image info
        coco_data['images'].append({
            'id': image_id,
            'file_name': img_file,
            'width': img_width,
            'height': img_height
        })

        # Parse label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            annotations = parse_yolo_label(label_path, img_width, img_height)

            # Add annotations with proper IDs
            for ann in annotations:
                ann['id'] = annotation_id
                ann['image_id'] = image_id
                coco_data['annotations'].append(ann)
                annotation_id += 1

        image_id += 1

    # Write COCO JSON file (Coconuts-1 style: _annotations.coco.json)
    coco_json_path = os.path.join(coco_split_dir, '_annotations.coco.json')
    with open(coco_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Created: {coco_json_path}")
    print(f"    Images: {len(coco_data['images'])}")
    print(f"    Annotations: {len(coco_data['annotations'])}")

    return len(coco_data['images']), len(coco_data['annotations'])


def convert_yolo_to_coco(yolo_dir: str, coco_dir: str) -> None:
    """
    Convert complete YOLO dataset to COCO format.

    Args:
        yolo_dir: Path to YOLO dataset directory
        coco_dir: Path to output COCO dataset directory
    """
    print("=" * 60)
    print("YOLO to COCO Segmentation Converter")
    print("=" * 60)
    print(f"Input:  {yolo_dir}")
    print(f"Output: {coco_dir}")
    print("=" * 60)

    # Load class names
    print("\n[1/4] Loading class names...")
    class_names = load_class_names(yolo_dir)
    categories = create_categories(class_names)
    print(f"  [OK] Loaded {len(class_names)} classes")

    # Create output directory
    os.makedirs(coco_dir, exist_ok=True)

    # Copy config file
    config_src = os.path.join(yolo_dir, 'config_used.yaml')
    if os.path.exists(config_src):
        config_dst = os.path.join(coco_dir, 'config_used.yaml')
        shutil.copy2(config_src, config_dst)
        print(f"  [OK] Copied config: {config_dst}")

    # Convert each split
    splits = [
        ('train', 'train'),
        ('val', 'valid'),  # YOLO uses 'val', COCO uses 'valid'
        ('test', 'test')
    ]

    total_images = 0
    total_annotations = 0

    for yolo_split, coco_split in splits:
        yolo_split_dir = os.path.join(yolo_dir, yolo_split)

        if not os.path.exists(yolo_split_dir):
            print(f"\n[SKIP] {yolo_split} split not found")
            continue

        print(f"\n[{splits.index((yolo_split, coco_split)) + 2}/4] Converting {yolo_split.upper()} split...")

        coco_split_dir = os.path.join(coco_dir, coco_split)

        img_count, ann_count = convert_split(
            yolo_split_dir,
            coco_split_dir,
            coco_split,
            categories
        )

        total_images += img_count
        total_annotations += ann_count

    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Total Images:      {total_images}")
    print(f"Total Annotations: {total_annotations}")
    print(f"Output Directory:  {coco_dir}")
    print("=" * 60)

    # Verify output structure
    print("\nOutput Structure:")
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(coco_dir, split)
        if os.path.exists(split_dir):
            json_file = os.path.join(split_dir, '_annotations.coco.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                print(f"  {split}/")
                print(f"    ├── _annotations.coco.json ({len(data['images'])} images, {len(data['annotations'])} annotations)")
                print(f"    └── *.png ({len(data['images'])} files)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO segmentation dataset to COCO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert yolo_seg_base to coco_seg_base
  python -m tools.convert_yolo_to_coco

  # Convert custom directories
  python -m tools.convert_yolo_to_coco --input my_yolo_dataset --output my_coco_dataset
        """
    )

    parser.add_argument(
        '--input',
        default='yolo_seg_base',
        help='Input YOLO dataset directory (default: yolo_seg_base)'
    )

    parser.add_argument(
        '--output',
        default='coco_seg_base',
        help='Output COCO dataset directory (default: coco_seg_base)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return 1

    # Check for required files
    classes_file = os.path.join(args.input, 'classes.txt')
    if not os.path.exists(classes_file):
        print(f"Error: classes.txt not found in {args.input}")
        return 1

    # Run conversion
    convert_yolo_to_coco(args.input, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
