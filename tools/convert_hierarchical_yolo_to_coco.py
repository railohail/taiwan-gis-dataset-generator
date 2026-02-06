"""
Hierarchical YOLO to COCO Format Converter for H-DETR

Converts hierarchical YOLO segmentation dataset (with L0/L1 levels) to
hierarchical COCO format as specified in model.md for H-DETR training.

Input structure (Hierarchical YOLO):
    yolo_hierarchical/
    ├── classes_l0.txt          # County class names
    ├── classes_l1.txt          # Township class names
    ├── hierarchy.json          # Parent-child mappings
    ├── train/
    │   ├── images/             # PNG images
    │   ├── labels/             # *_l0.txt, *_l1.txt per image
    │   └── visualizations/
    └── val/
        └── ...

Output structure (Hierarchical COCO):
    coco_hierarchical/
    ├── train/
    │   ├── *.png               # Images
    │   └── _annotations.coco.json
    ├── valid/
    │   └── ...
    └── hierarchy_info.json     # Category mappings

Usage:
    python -m tools.convert_hierarchical_yolo_to_coco
    python -m tools.convert_hierarchical_yolo_to_coco --input yolo_hierarchical_test --output coco_hierarchical_test

"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Any, Optional


def load_class_names(yolo_dir: str, level: int) -> List[str]:
    """Load class names from classes_l0.txt or classes_l1.txt."""
    classes_path = os.path.join(yolo_dir, f'classes_l{level}.txt')
    if not os.path.exists(classes_path):
        # Fallback to classes.txt for non-hierarchical
        classes_path = os.path.join(yolo_dir, 'classes.txt')

    with open(classes_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def load_hierarchy_data(yolo_dir: str) -> Optional[Dict[str, Any]]:
    """Load hierarchy.json containing parent-child mappings."""
    hierarchy_path = os.path.join(yolo_dir, 'hierarchy.json')
    if not os.path.exists(hierarchy_path):
        return None

    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_hierarchical_categories(
    class_names_l0: List[str],
    class_names_l1: List[str],
    hierarchy_data: Optional[Dict[str, Any]]
) -> Tuple[List[Dict], List[Dict], Dict[str, List[int]], Dict[int, int]]:
    """
    Create COCO categories for both levels with parent-child relationships.

    Returns:
        categories_l0: List of county categories
        categories_l1: List of township categories with parent info
        hierarchy_map: Dict mapping county_id -> list of township_ids
        township_parent_map: Dict mapping township_class_id -> parent_county_class_id
    """
    categories_l0 = []
    categories_l1 = []
    hierarchy_map = {}  # county_class_id -> [township_class_ids]
    township_parent_map = {}  # township_class_id -> parent_county_class_id

    # Build lookup from hierarchy data
    county_english = {}
    township_english = {}
    township_parent_class = {}

    if hierarchy_data:
        # Extract county info
        for code, info in hierarchy_data.get('counties', {}).items():
            class_id = info.get('class_id', 0)
            county_english[class_id] = info.get('english', f'County_{class_id}')
            hierarchy_map[class_id] = []

        # Extract township info
        for town_id, info in hierarchy_data.get('townships', {}).items():
            class_id = info.get('class_id', 0)
            township_english[class_id] = info.get('english', f'Township_{class_id}')
            # Try multiple possible key names for parent class
            parent_class = info.get('parent_class') or info.get('parent_class_id')
            if parent_class is not None:
                township_parent_class[class_id] = parent_class
                township_parent_map[class_id] = parent_class
                if parent_class in hierarchy_map:
                    hierarchy_map[parent_class].append(class_id)

    # Create L0 (County) categories
    for idx, name in enumerate(class_names_l0):
        english = county_english.get(idx, name)
        categories_l0.append({
            'id': idx,
            'name': name,
            'english': english,
            'supercategory': 'county'
        })

    # Create L1 (Township) categories with parent info
    for idx, name in enumerate(class_names_l1):
        english = township_english.get(idx, name)
        parent_class = township_parent_class.get(idx)

        cat = {
            'id': idx,
            'name': name,
            'english': english,
            'supercategory': 'township'
        }

        if parent_class is not None:
            cat['parent_id'] = parent_class
            cat['parent_class'] = parent_class

        categories_l1.append(cat)

    return categories_l0, categories_l1, hierarchy_map, township_parent_map


def yolo_segmentation_to_coco(yolo_coords: List[float], img_width: int, img_height: int) -> List[float]:
    """Convert YOLO normalized segmentation to COCO pixel coordinates."""
    coco_seg = []
    for i in range(0, len(yolo_coords), 2):
        x_norm = yolo_coords[i]
        y_norm = yolo_coords[i + 1]
        px = x_norm * img_width
        py = y_norm * img_height
        coco_seg.extend([px, py])
    return coco_seg


def calculate_bbox_from_segmentation(segmentation: List[float]) -> List[float]:
    """Calculate bounding box [x, y, w, h] from segmentation coordinates."""
    if not segmentation:
        return [0, 0, 0, 0]

    x_coords = segmentation[0::2]
    y_coords = segmentation[1::2]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def calculate_area_from_segmentation(segmentation: List[float]) -> float:
    """Calculate area using Shoelace formula."""
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


def parse_yolo_label(label_path: str, img_width: int, img_height: int) -> List[Dict]:
    """Parse YOLO label file and convert to COCO annotations."""
    annotations = []

    if not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + at least 3 points (x,y)
            continue

        class_id = int(parts[0])
        yolo_coords = [float(x) for x in parts[1:]]

        coco_seg = yolo_segmentation_to_coco(yolo_coords, img_width, img_height)

        if len(coco_seg) < 6:
            continue

        bbox = calculate_bbox_from_segmentation(coco_seg)
        area = calculate_area_from_segmentation(coco_seg)

        if area < 10:
            continue

        annotations.append({
            'category_id': class_id,
            'segmentation': [coco_seg],
            'bbox': bbox,
            'area': area,
            'iscrowd': 0
        })

    return annotations


def convert_hierarchical_split(
    yolo_split_dir: str,
    coco_split_dir: str,
    split_name: str,
    categories_l0: List[Dict],
    categories_l1: List[Dict],
    hierarchy_map: Dict[str, List[int]],
    township_parent_map: Dict[int, int]
) -> Tuple[int, int, int]:
    """
    Convert one split from hierarchical YOLO to hierarchical COCO format.

    Returns:
        (images_count, l0_annotations_count, l1_annotations_count)
    """
    images_dir = os.path.join(yolo_split_dir, 'images')
    labels_dir = os.path.join(yolo_split_dir, 'labels')

    os.makedirs(coco_split_dir, exist_ok=True)

    # Copy visualizations if they exist
    vis_src = os.path.join(yolo_split_dir, 'visualizations')
    if os.path.exists(vis_src):
        vis_dst = os.path.join(os.path.dirname(coco_split_dir), 'visualizations', split_name)
        os.makedirs(vis_dst, exist_ok=True)
        print(f"  Copying visualizations...")
        for vis_file in os.listdir(vis_src):
            shutil.copy2(os.path.join(vis_src, vis_file), os.path.join(vis_dst, vis_file))

    # Initialize hierarchical COCO structure
    coco_data = {
        "info": {
            "description": f"Taiwan Administrative Boundaries - Hierarchical ({split_name.upper()})",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Hierarchical YOLO to COCO Converter",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hierarchy_levels": ["county", "township"]
        },
        "licenses": [{"url": "", "id": 1, "name": "Unknown"}],
        "images": [],
        "categories_l0": categories_l0,
        "categories_l1": categories_l1,
        "hierarchy": {str(k): v for k, v in hierarchy_map.items()},
        "annotations": []  # Combined annotations with category_level field
    }

    # Get all image files
    if not os.path.exists(images_dir):
        print(f"  WARNING: Images directory not found: {images_dir}")
        return 0, 0, 0

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    print(f"  Processing {len(image_files)} images...")

    image_id = 0
    annotation_id = 0
    l0_count = 0
    l1_count = 0

    for img_file in tqdm(image_files, desc=f"  {split_name}", unit="img"):
        # Copy image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(coco_split_dir, img_file)
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

        # Base name for label files
        base_name = os.path.splitext(img_file)[0]
        # Remove noise suffix if present to get the base
        # e.g., hier_10005_001_1_modified_noise_clean -> hier_10005_001_1_modified_noise_clean

        # Parse L0 labels
        l0_label_path = os.path.join(labels_dir, f"{base_name}_l0.txt")
        l0_annotations = parse_yolo_label(l0_label_path, img_width, img_height)

        # Track which counties are in this image for children linking
        county_ann_ids = {}  # county_class_id -> annotation_id

        for ann in l0_annotations:
            ann['id'] = annotation_id
            ann['image_id'] = image_id
            ann['category_level'] = 0
            ann['children'] = []  # Will be populated with L1 annotation IDs

            county_ann_ids[ann['category_id']] = annotation_id

            coco_data['annotations'].append(ann)
            annotation_id += 1
            l0_count += 1

        # Parse L1 labels
        l1_label_path = os.path.join(labels_dir, f"{base_name}_l1.txt")
        l1_annotations = parse_yolo_label(l1_label_path, img_width, img_height)

        for ann in l1_annotations:
            ann['id'] = annotation_id
            ann['image_id'] = image_id
            ann['category_level'] = 1

            # Add parent information
            parent_class = township_parent_map.get(ann['category_id'])
            if parent_class is not None:
                ann['parent_class'] = parent_class

                # Link to parent annotation if it exists
                if parent_class in county_ann_ids:
                    parent_ann_id = county_ann_ids[parent_class]
                    ann['parent_id'] = parent_ann_id

                    # Add this annotation ID to parent's children
                    for parent_ann in coco_data['annotations']:
                        if parent_ann['id'] == parent_ann_id:
                            parent_ann['children'].append(annotation_id)
                            break

            coco_data['annotations'].append(ann)
            annotation_id += 1
            l1_count += 1

        image_id += 1

    # Write COCO JSON file
    coco_json_path = os.path.join(coco_split_dir, '_annotations.coco.json')
    with open(coco_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print(f"  Created: {coco_json_path}")
    print(f"    Images: {len(coco_data['images'])}")
    print(f"    L0 (County) annotations: {l0_count}")
    print(f"    L1 (Township) annotations: {l1_count}")

    return len(coco_data['images']), l0_count, l1_count


def convert_hierarchical_yolo_to_coco(yolo_dir: str, coco_dir: str) -> None:
    """Convert complete hierarchical YOLO dataset to hierarchical COCO format."""
    print("=" * 70)
    print("Hierarchical YOLO to COCO Converter (H-DETR Format)")
    print("=" * 70)
    print(f"Input:  {yolo_dir}")
    print(f"Output: {coco_dir}")
    print("=" * 70)

    # Load class names for both levels
    print("\n[1/4] Loading class names...")
    class_names_l0 = load_class_names(yolo_dir, 0)
    class_names_l1 = load_class_names(yolo_dir, 1)
    print(f"  L0 (County): {len(class_names_l0)} classes")
    print(f"  L1 (Township): {len(class_names_l1)} classes")

    # Load hierarchy data
    print("\n[2/4] Loading hierarchy data...")
    hierarchy_data = load_hierarchy_data(yolo_dir)
    if hierarchy_data:
        print(f"  Loaded hierarchy.json")
        print(f"    Counties: {hierarchy_data.get('metadata', {}).get('num_counties', '?')}")
        print(f"    Townships: {hierarchy_data.get('metadata', {}).get('num_townships', '?')}")
    else:
        print("  WARNING: hierarchy.json not found, parent-child links will be missing")

    # Create categories with hierarchy
    categories_l0, categories_l1, hierarchy_map, township_parent_map = create_hierarchical_categories(
        class_names_l0, class_names_l1, hierarchy_data
    )

    print(f"  Township parent mappings: {len(township_parent_map)} townships with parent info")

    # Create output directory
    os.makedirs(coco_dir, exist_ok=True)

    # Save hierarchy info
    hierarchy_info = {
        'categories_l0': categories_l0,
        'categories_l1': categories_l1,
        'hierarchy': {str(k): v for k, v in hierarchy_map.items()},
        'num_counties': len(categories_l0),
        'num_townships': len(categories_l1)
    }
    hierarchy_info_path = os.path.join(coco_dir, 'hierarchy_info.json')
    with open(hierarchy_info_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy_info, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {hierarchy_info_path}")

    # Copy config if exists
    for config_file in ['config_used.yaml', 'dataset.yaml']:
        config_src = os.path.join(yolo_dir, config_file)
        if os.path.exists(config_src):
            config_dst = os.path.join(coco_dir, config_file)
            shutil.copy2(config_src, config_dst)
            print(f"  Copied: {config_file}")

    # Convert each split
    splits = [
        ('train', 'train'),
        ('val', 'valid'),
        ('test', 'test')
    ]

    total_images = 0
    total_l0 = 0
    total_l1 = 0

    step = 3
    for yolo_split, coco_split in splits:
        yolo_split_dir = os.path.join(yolo_dir, yolo_split)

        if not os.path.exists(yolo_split_dir):
            print(f"\n[{step}/4] SKIP: {yolo_split} split not found")
            step += 1
            continue

        print(f"\n[{step}/4] Converting {yolo_split.upper()} split...")

        coco_split_dir = os.path.join(coco_dir, coco_split)

        img_count, l0_count, l1_count = convert_hierarchical_split(
            yolo_split_dir,
            coco_split_dir,
            coco_split,
            categories_l0,
            categories_l1,
            hierarchy_map,
            township_parent_map
        )

        total_images += img_count
        total_l0 += l0_count
        total_l1 += l1_count
        step += 1

    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"Total Images:              {total_images}")
    print(f"Total L0 (County) Anns:    {total_l0}")
    print(f"Total L1 (Township) Anns:  {total_l1}")
    print(f"Output Directory:          {coco_dir}")
    print("=" * 70)

    # Verify output structure
    print("\nOutput Structure:")
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(coco_dir, split)
        if os.path.exists(split_dir):
            json_file = os.path.join(split_dir, '_annotations.coco.json')
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                l0_anns = sum(1 for a in data['annotations'] if a.get('category_level') == 0)
                l1_anns = sum(1 for a in data['annotations'] if a.get('category_level') == 1)

                print(f"  {split}/")
                print(f"    ├── _annotations.coco.json")
                print(f"    │   ├── images: {len(data['images'])}")
                print(f"    │   ├── L0 annotations: {l0_anns}")
                print(f"    │   └── L1 annotations: {l1_anns}")
                print(f"    └── *.png ({len(data['images'])} files)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert hierarchical YOLO segmentation dataset to COCO format for H-DETR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert yolo_hierarchical_test to coco_hierarchical_test
  python -m tools.convert_hierarchical_yolo_to_coco --input yolo_hierarchical_test --output coco_hierarchical_test

  # Use default directories
  python -m tools.convert_hierarchical_yolo_to_coco
        """
    )

    parser.add_argument(
        '--input',
        default='yolo_hierarchical_test',
        help='Input hierarchical YOLO dataset directory (default: yolo_hierarchical_test)'
    )

    parser.add_argument(
        '--output',
        default='coco_hierarchical_test',
        help='Output hierarchical COCO dataset directory (default: coco_hierarchical_test)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return 1

    # Check for required files
    l0_classes = os.path.join(args.input, 'classes_l0.txt')
    l1_classes = os.path.join(args.input, 'classes_l1.txt')

    if not os.path.exists(l0_classes):
        print(f"Error: classes_l0.txt not found in {args.input}")
        return 1

    if not os.path.exists(l1_classes):
        print(f"Error: classes_l1.txt not found in {args.input}")
        return 1

    # Run conversion
    convert_hierarchical_yolo_to_coco(args.input, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
