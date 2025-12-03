"""
Dataset format writers.

This package contains format-specific writers for exporting annotations:
- yolo: YOLO format (.txt files with normalized coordinates)

Future formats can be added as separate modules (e.g., voc.py for Pascal VOC).
"""

from .yolo import (
    coco_to_yolo_bbox,
    coco_segmentation_to_yolo,
    write_yolo_annotation_file,
    initialize_yolo_dataset,
    batch_write_yolo_annotations,
    batch_append_to_yolo_buffer,
    flush_yolo_batch,
    reset_yolo_batch,
    set_yolo_batch_size,
    initialize_split_manager,
    get_split_for_file,
    get_split_directories,
    print_split_statistics,
)

__all__ = [
    # YOLO format
    'coco_to_yolo_bbox',
    'coco_segmentation_to_yolo',
    'write_yolo_annotation_file',
    'initialize_yolo_dataset',
    'batch_write_yolo_annotations',
    'batch_append_to_yolo_buffer',
    'flush_yolo_batch',
    'reset_yolo_batch',
    'set_yolo_batch_size',
    'initialize_split_manager',
    'get_split_for_file',
    'get_split_directories',
    'print_split_statistics',
]
