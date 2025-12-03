"""
Taiwan GIS Dataset Generator - Tools Package

This package contains standalone utility tools for dataset manipulation
and interactive mask region selection.

Tools:
    clean_mask           - Interactive mask region selector for excluding
                          zoom-ins, legends, and decorative elements
    convert_yolo_to_coco - Convert YOLO segmentation datasets to COCO format

Usage:
    python -m tools.clean_mask
    python -m tools.convert_yolo_to_coco --input yolo_seg_base --output coco_seg_base
"""

__all__ = ['clean_mask', 'convert_yolo_to_coco']
