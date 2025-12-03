"""
GIS Dataset Generator Utilities Package.

This package provides modular components for converting Taiwan GIS map data
into machine learning datasets (COCO, YOLO formats).

Module Structure:
    config.py      - Configuration, county mappings, YAML loading
    annotations.py - Annotation creation and COCO JSON handling
    geometry.py    - GIS/shapefile operations, coordinate transforms
    image.py       - Image processing, augmentation, noise generation
    pipeline.py    - Main dataset generation workflow
    visualization.py - Debug overlays and verification utilities
    masks.py       - Mask region database for exclusion areas

    writers/       - Format-specific annotation writers
        yolo.py    - YOLO format output

    core/          - Core infrastructure
        config.py  - Dataclass-based configuration
        constants.py - Enums and constant values
        logger.py  - Logging utilities

    models/        - Data models
        annotations.py - Annotation dataclasses
        image_info.py  - Image metadata models

"""

# =============================================================================
# New module names (preferred)
# =============================================================================

from .config import (
    load_config,
    create_categories,
    COUNTY_ENGLISH_NAMES,
    COUNTY_TO_CLASS,
    normalize_county_name,
)

from .annotations import (
    initialize_coco_file,
    create_annotations_for_image,
    create_annotations_for_window,
    append_to_coco_file,
    batch_append_to_coco_buffer,
    flush_coco_batch,
    set_batch_size,
)

from .geometry import (
    find_tif_files,
    load_data,
    extract_single_district_image,
    crop_image_and_shapefile,
)

from .image import (
    preprocess_raster,
    convert_to_grayscale,
    unicode_safe_imwrite,
)

from .pipeline import (
    process_separate_districts,
    process_combined_maps,
    reset_processing_state,
)

from .masks import (
    MaskDatabase,
    get_mask_database,
    filter_annotations_with_masks,
)

# =============================================================================
# YOLO writer (from writers subpackage)
# =============================================================================

from .writers.yolo import (
    initialize_yolo_dataset,
    set_yolo_batch_size,
    initialize_split_manager,
    print_split_statistics,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Configuration
    'load_config',
    'create_categories',
    'COUNTY_ENGLISH_NAMES',
    'COUNTY_TO_CLASS',
    'normalize_county_name',

    # Annotations
    'initialize_coco_file',
    'create_annotations_for_image',
    'create_annotations_for_window',
    'append_to_coco_file',
    'batch_append_to_coco_buffer',
    'flush_coco_batch',
    'set_batch_size',

    # Geometry
    'find_tif_files',
    'load_data',
    'extract_single_district_image',
    'crop_image_and_shapefile',

    # Image Processing
    'preprocess_raster',
    'convert_to_grayscale',
    'unicode_safe_imwrite',

    # Pipeline
    'process_separate_districts',
    'process_combined_maps',
    'reset_processing_state',

    # Masks
    'MaskDatabase',
    'get_mask_database',
    'filter_annotations_with_masks',

    # YOLO
    'initialize_yolo_dataset',
    'set_yolo_batch_size',
    'initialize_split_manager',
    'print_split_statistics',
]
