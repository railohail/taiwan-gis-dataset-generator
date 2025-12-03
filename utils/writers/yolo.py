"""
YOLO format annotation writer.

This module handles:
- COCO to YOLO bbox/segmentation conversion
- YOLO annotation file writing (.txt format)
- Dataset structure initialization (classes.txt, dataset.yaml)
- Train/val/test split management
- Batch buffered writing for performance

YOLO formats:
- Bbox: class_id x_center y_center width height (normalized 0-1)
- Segmentation: class_id x1 y1 x2 y2 ... xn yn (normalized 0-1)
"""

import os
import json
import random
from datetime import datetime


def coco_to_yolo_bbox(coco_bbox, image_width, image_height):
    """
    Convert COCO bbox format to YOLO format.

    COCO format: [x, y, width, height] (absolute pixels, top-left corner)
    YOLO format: [x_center, y_center, width, height] (normalized 0-1, center)

    Args:
        coco_bbox: [x, y, w, h] in pixels
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        [x_center, y_center, width, height] normalized to 0-1
    """
    x, y, w, h = coco_bbox

    # Calculate center point
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height

    # Normalize width and height
    width = w / image_width
    height = h / image_height

    # Clamp to [0, 1] range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return [x_center, y_center, width, height]


def coco_segmentation_to_yolo(segmentation, image_width, image_height):
    """
    Convert COCO segmentation to YOLO segmentation format.

    COCO format: [[x1, y1, x2, y2, ..., xn, yn]] (absolute pixels)
    YOLO format: [x1, y1, x2, y2, ..., xn, yn] (normalized 0-1)

    Args:
        segmentation: COCO segmentation (list of lists)
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        List of normalized coordinates [x1, y1, x2, y2, ...]
    """
    if not segmentation or len(segmentation) == 0:
        return []

    # Get the first polygon (COCO segmentation is list of polygons)
    polygon = segmentation[0]

    # Normalize all coordinates
    normalized = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / image_width
        y = polygon[i + 1] / image_height
        # Clamp to [0, 1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        normalized.extend([x, y])

    return normalized


def write_yolo_annotation_file(image_name, annotations, image_width, image_height, output_dir, annotation_type='bbox'):
    """
    Write YOLO format annotation file for a single image.

    YOLO bbox format: Each line is "class_id x_center y_center width height"
    YOLO segmentation format: Each line is "class_id x1 y1 x2 y2 ... xn yn"
    File naming: same as image but with .txt extension

    Args:
        image_name: Name of the image file (e.g., "separate_taipei_clean.png")
        annotations: List of COCO annotations for this image
        image_width: Image width in pixels
        image_height: Image height in pixels
        output_dir: Directory to save annotation files
        annotation_type: 'bbox' or 'segmentation'

    Returns:
        Path to the saved annotation file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get annotation filename (same as image but .txt)
    base_name = os.path.splitext(image_name)[0]
    annotation_filename = f"{base_name}.txt"
    annotation_path = os.path.join(output_dir, annotation_filename)

    # Write YOLO format annotations
    with open(annotation_path, 'w', encoding='utf-8') as f:
        for ann in annotations:
            class_id = ann['category_id']

            if annotation_type == 'segmentation':
                # YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
                if 'segmentation' in ann and ann['segmentation']:
                    yolo_seg = coco_segmentation_to_yolo(ann['segmentation'], image_width, image_height)
                    if yolo_seg and len(yolo_seg) >= 6:  # At least 3 points
                        coords_str = ' '.join([f"{coord:.6f}" for coord in yolo_seg])
                        line = f"{class_id} {coords_str}\n"
                        f.write(line)
            else:
                # YOLO bbox format: class_id x_center y_center width height
                coco_bbox = ann['bbox']
                yolo_bbox = coco_to_yolo_bbox(coco_bbox, image_width, image_height)
                line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                f.write(line)

    return annotation_path


def initialize_yolo_dataset(base_output_dir, categories, use_split=True):
    """
    Initialize YOLO dataset structure and create classes file.

    YOLO dataset structure (with split=True):
    - train/
        - images/     : Training images
        - labels/     : Training annotations
    - val/
        - images/     : Validation images
        - labels/     : Validation annotations
    - test/ (optional)
        - images/     : Test images
        - labels/     : Test annotations
    - classes.txt     : List of class names (one per line, order = class_id)
    - dataset.yaml    : Dataset configuration for training

    YOLO dataset structure (with split=False, legacy):
    - images/         : All images
    - labels/         : Annotation txt files (one per image)
    - classes.txt
    - dataset.yaml

    Args:
        base_output_dir: Base output directory
        categories: List of category dictionaries from COCO format
        use_split: If True, create train/val/test split structure (default: True)

    Returns:
        Tuple of (images_dir, labels_dir, classes_file_path, dataset_yaml_path)
    """
    # Create directory structure
    if use_split:
        # Create train/val/test split structure
        train_images_dir = os.path.join(base_output_dir, 'train', 'images')
        train_labels_dir = os.path.join(base_output_dir, 'train', 'labels')
        val_images_dir = os.path.join(base_output_dir, 'val', 'images')
        val_labels_dir = os.path.join(base_output_dir, 'val', 'labels')
        test_images_dir = os.path.join(base_output_dir, 'test', 'images')
        test_labels_dir = os.path.join(base_output_dir, 'test', 'labels')

        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        os.makedirs(test_images_dir, exist_ok=True)
        os.makedirs(test_labels_dir, exist_ok=True)

        # For compatibility, return train directories as default
        images_dir = train_images_dir
        labels_dir = train_labels_dir
    else:
        # Legacy: single images/labels directory
        images_dir = os.path.join(base_output_dir, 'images')
        labels_dir = os.path.join(base_output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    # Write classes.txt (one class name per line, order matters)
    classes_file_path = os.path.join(base_output_dir, 'classes.txt')
    with open(classes_file_path, 'w', encoding='utf-8') as f:
        # Sort by ID to ensure correct order
        sorted_categories = sorted(categories, key=lambda x: x['id'])
        for cat in sorted_categories:
            f.write(f"{cat['name']}\n")

    print(f"Created YOLO classes file: {classes_file_path} ({len(categories)} classes)")

    # Write dataset.yaml for YOLO training
    dataset_yaml_path = os.path.join(base_output_dir, 'dataset.yaml')
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        f.write("# YOLO Dataset Configuration\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("# Paths (relative to this file)\n")
        f.write("path: .  # dataset root dir\n")

        if use_split:
            f.write("train: train/images  # train images (relative to 'path')\n")
            f.write("val: val/images      # val images (relative to 'path')\n")
            f.write("test: test/images    # test images (relative to 'path', optional)\n\n")
        else:
            f.write("train: images  # train images (relative to 'path')\n")
            f.write("val: images    # val images (relative to 'path')\n\n")

        f.write("# Classes\n")
        f.write(f"nc: {len(categories)}  # number of classes\n")
        f.write("names:\n")

        # Sort by ID to ensure correct order
        sorted_categories = sorted(categories, key=lambda x: x['id'])
        for cat in sorted_categories:
            f.write(f"  {cat['id']}: {cat['name']}\n")

    print(f"Created YOLO dataset config: {dataset_yaml_path}")

    return images_dir, labels_dir, classes_file_path, dataset_yaml_path


def batch_write_yolo_annotations(images_info, annotations_by_image_id, labels_dir, annotation_type='bbox'):
    """
    Write YOLO annotation files for multiple images in batch.

    Args:
        images_info: List of image info dictionaries (id, width, height, file_name)
        annotations_by_image_id: Dict mapping image_id to list of annotations
        labels_dir: Directory to save label files
        annotation_type: 'bbox' or 'segmentation'

    Returns:
        Number of annotation files written
    """
    count = 0
    for image_info in images_info:
        image_id = image_info['id']
        image_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        # Get annotations for this image
        annotations = annotations_by_image_id.get(image_id, [])

        # Write YOLO annotation file
        write_yolo_annotation_file(image_name, annotations, image_width, image_height, labels_dir, annotation_type)
        count += 1

    return count


# YOLO batch buffer for efficient writing
_yolo_batch_buffer = {'images': [], 'annotations': {}}
_yolo_batch_size = 100  # Default batch size


def set_yolo_batch_size(size):
    """Set the batch size for YOLO buffered writes."""
    global _yolo_batch_size
    _yolo_batch_size = size


def batch_append_to_yolo_buffer(image_info, annotations):
    """
    Add image and annotations to YOLO batch buffer.

    Args:
        image_info: Single image info dict
        annotations: List of annotations for this image
    """
    global _yolo_batch_buffer

    image_id = image_info['id']
    _yolo_batch_buffer['images'].append(image_info)
    _yolo_batch_buffer['annotations'][image_id] = annotations


def flush_yolo_batch(labels_dir, annotation_type='bbox', force=False):
    """
    Flush YOLO batch buffer to disk.

    Args:
        labels_dir: Directory to save label files
        annotation_type: 'bbox' or 'segmentation'
        force: If True, flush regardless of buffer size

    Returns:
        Number of files written
    """
    global _yolo_batch_buffer

    if not _yolo_batch_buffer['images'] and not force:
        return 0

    if len(_yolo_batch_buffer['images']) < _yolo_batch_size and not force:
        return 0

    try:
        # Write all buffered annotations
        count = batch_write_yolo_annotations(
            _yolo_batch_buffer['images'],
            _yolo_batch_buffer['annotations'],
            labels_dir,
            annotation_type
        )

        print(f"    YOLO BATCH FLUSH: Wrote {count} annotation files ({annotation_type} format)")

        # Clear buffer
        _yolo_batch_buffer['images'].clear()
        _yolo_batch_buffer['annotations'].clear()

        return count

    except Exception as e:
        print(f"    ERROR in flush_yolo_batch: {e}")
        raise


def reset_yolo_batch():
    """Reset the YOLO batch buffer."""
    global _yolo_batch_buffer
    _yolo_batch_buffer['images'].clear()
    _yolo_batch_buffer['annotations'].clear()


# Dataset split manager for train/val/test
_split_state = {
    'enabled': False,
    'train_ratio': 0.7,
    'val_ratio': 0.2,
    'test_ratio': 0.1,
    'base_dir': None,
    'seed': 42,
    'train_files': set(),
    'val_files': set(),
    'test_files': set(),
    'file_to_split': {}  # Maps file path to split name
}


def initialize_split_manager(base_output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42, file_list=None):
    """
    Initialize the dataset split manager for train/val/test splits.

    DETERMINISTIC SPLIT: Pre-assigns files to splits to guarantee exact distribution.
    This replaces the old probability-based approach which could lead to imbalanced splits.

    Args:
        base_output_dir: Base output directory
        train_ratio: Proportion for training set (default: 0.7 = 70%)
        val_ratio: Proportion for validation set (default: 0.2 = 20%)
        test_ratio: Proportion for test set (default: 0.1 = 10%)
        seed: Random seed for reproducibility
        file_list: Optional list of file paths to pre-assign to splits.
                   If provided, files will be deterministically assigned.
                   If None, falls back to per-file lookup mode.
    """
    global _split_state

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:  # Relaxed tolerance to 1%
        raise ValueError(f"Split ratios must sum to 1.0 (±1%), got {total:.3f}")

    _split_state['enabled'] = True
    _split_state['train_ratio'] = train_ratio
    _split_state['val_ratio'] = val_ratio
    _split_state['test_ratio'] = test_ratio
    _split_state['base_dir'] = base_output_dir
    _split_state['seed'] = seed

    # Deterministic pre-assignment if file list provided
    if file_list:
        random.seed(seed)
        shuffled = list(file_list)
        random.shuffle(shuffled)

        n_total = len(shuffled)

        # Use round() instead of int() to handle small datasets better
        # Special case: if ratio is 0.0, force count to 0
        if train_ratio == 0.0:
            n_train = 0
        else:
            n_train = max(1, round(n_total * train_ratio)) if train_ratio > 0 else 0

        if val_ratio == 0.0:
            n_val = 0
        else:
            n_val = max(1, round(n_total * val_ratio)) if val_ratio > 0 else 0

        # n_test gets the remainder to ensure all files are assigned
        # But if test_ratio is 0.0, we need to redistribute remainder to train/val
        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train + n_val]
        test_files = shuffled[n_train + n_val:]

        # If test_ratio is 0.0 but we have remainder files, add them to train
        if test_ratio == 0.0 and len(test_files) > 0:
            train_files.extend(test_files)
            test_files = []

        _split_state['train_files'] = set(train_files)
        _split_state['val_files'] = set(val_files)
        _split_state['test_files'] = set(test_files)

        # Create lookup dictionary for O(1) access
        _split_state['file_to_split'] = {}
        for f in train_files:
            _split_state['file_to_split'][f] = 'train'
        for f in val_files:
            _split_state['file_to_split'][f] = 'val'
        for f in test_files:
            _split_state['file_to_split'][f] = 'test'

        print(f"Dataset split initialized (DETERMINISTIC):")
        print(f"  Train: {len(train_files)} files ({len(train_files)/n_total*100:.1f}%)")
        print(f"  Val:   {len(val_files)} files ({len(val_files)/n_total*100:.1f}%)")
        print(f"  Test:  {len(test_files)} files ({len(test_files)/n_total*100:.1f}%)")
    else:
        # No file list provided - clear pre-assignments
        _split_state['train_files'] = set()
        _split_state['val_files'] = set()
        _split_state['test_files'] = set()
        _split_state['file_to_split'] = {}
        print(f"Dataset split initialized: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
        print("  WARNING: No file list provided. Using per-file lookup mode.")


def get_split_for_file(file_path):
    """
    Determine which split (train/val/test) a file should go into.

    NEW: Uses deterministic pre-assignment if available, ensuring exact ratio distribution.
    Falls back to warning if file not found in pre-assignment.

    Args:
        file_path: Path to the file (should match what was passed to initialize_split_manager)

    Returns:
        str: 'train', 'val', or 'test'
    """
    global _split_state

    if not _split_state['enabled']:
        return 'train'  # Default to train if splitting is disabled

    # Check if we have pre-assigned splits
    if _split_state['file_to_split']:
        # Deterministic lookup
        split = _split_state['file_to_split'].get(file_path)
        if split:
            return split
        else:
            # File not in pre-assignment - this shouldn't happen
            print(f"  WARNING: File '{file_path}' not found in pre-assigned splits. Defaulting to train.")
            return 'train'
    else:
        # Fallback: No pre-assignment available
        print(f"  WARNING: No pre-assignment available. File '{file_path}' defaulting to train.")
        return 'train'


def get_split_for_image():
    """
    DEPRECATED: Use get_split_for_file() instead for deterministic splits.

    This function remains for backward compatibility but will issue a warning.
    It uses random assignment which does NOT guarantee exact ratio distribution.

    Returns:
        str: 'train', 'val', or 'test'
    """
    global _split_state

    if not _split_state['enabled']:
        return 'train'

    print("  WARNING: get_split_for_image() is deprecated. Use get_split_for_file() for deterministic splits.")

    # Random assignment based on ratios (OLD BEHAVIOR)
    rand_val = random.random()

    if rand_val < _split_state['train_ratio']:
        return 'train'
    elif rand_val < _split_state['train_ratio'] + _split_state['val_ratio']:
        return 'val'
    else:
        return 'test'


def get_split_directories(base_output_dir, split_name, format_type='coco'):
    """
    Get the images and labels directories for a specific split.
    Creates directories if they don't exist.

    For COCO format (Coconuts-1 style):
        - Images saved directly in split folder (e.g., train/image1.jpg)
        - Returns split_dir as images_dir

    For YOLO format (original style):
        - Images in split/images/ subdirectory
        - Labels in split/labels/ subdirectory

    Args:
        base_output_dir: Base output directory
        split_name: 'train', 'val', 'valid', or 'test'
        format_type: 'coco' or 'yolo' (default: 'coco')

    Returns:
        Tuple of (images_dir, labels_dir)
    """
    # Normalize split name: 'val' -> 'valid' for COCO format consistency
    if format_type == 'coco' and split_name == 'val':
        split_name = 'valid'

    if format_type == 'coco':
        # Coconuts-1 style: flat structure with images directly in split folder
        split_dir = os.path.join(base_output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        return split_dir, None  # No separate labels dir for COCO
    else:
        # YOLO style: subdirectories for images and labels
        images_dir = os.path.join(base_output_dir, split_name, 'images')
        labels_dir = os.path.join(base_output_dir, split_name, 'labels')

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        return images_dir, labels_dir


def print_split_statistics(base_output_dir):
    """
    Print statistics about the train/val/test split.

    Args:
        base_output_dir: Base output directory
    """
    train_images_dir = os.path.join(base_output_dir, 'train', 'images')
    val_images_dir = os.path.join(base_output_dir, 'val', 'images')
    test_images_dir = os.path.join(base_output_dir, 'test', 'images')

    train_count = len([f for f in os.listdir(train_images_dir) if f.endswith('.png')]) if os.path.exists(train_images_dir) else 0
    val_count = len([f for f in os.listdir(val_images_dir) if f.endswith('.png')]) if os.path.exists(val_images_dir) else 0
    test_count = len([f for f in os.listdir(test_images_dir) if f.endswith('.png')]) if os.path.exists(test_images_dir) else 0

    total = train_count + val_count + test_count

    if total > 0:
        print("\n" + "=" * 60)
        print("DATASET SPLIT STATISTICS")
        print("=" * 60)
        print(f"  Train: {train_count:4d} images ({train_count/total*100:5.1f}%)")
        print(f"  Val:   {val_count:4d} images ({val_count/total*100:5.1f}%)")
        print(f"  Test:  {test_count:4d} images ({test_count/total*100:5.1f}%)")
        print(f"  Total: {total:4d} images")
        print("=" * 60)
    else:
        print("No images found in split directories.")
