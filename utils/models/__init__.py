"""
Data models for the GIS dataset generator.

This package contains dataclasses and type definitions for
annotations, image metadata, and other data structures.
"""

from .annotations import Annotation, BoundingBox
from .image_info import ImageInfo

__all__ = [
    "Annotation",
    "BoundingBox",
    "ImageInfo",
]
