# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GIS_GEN is a production-ready dataset generator for Taiwan county GIS map data. It converts TIF raster files with shapefile annotations into COCO or YOLO format datasets for machine learning tasks (instance segmentation, object detection).

## Commands

### Run Dataset Generation
```bash
# Default configuration
python main.py

# Custom configuration
python main.py --config configs/yolo-seg.yaml

# Verbose logging
python main.py --verbose
```

### Interactive Mask Tool
```bash
# Define exclusion regions (zoom-ins, legends) interactively
python -m tools.clean_mask
```

### YOLO to COCO Converter
```bash
# Convert YOLO dataset to COCO format
python -m tools.convert_yolo_to_coco --input yolo_seg_base --output coco_seg_base
```

## Architecture

### Entry Point
- [main.py](main.py) - `DatasetGenerator` class orchestrates the pipeline

### Core Pipeline Flow
1. Load config → 2. Setup outputs → 3. Process districts → 4. Apply augmentations → 5. Generate annotations → 6. Verify/merge

### Module Structure (`utils/`)

**Core Configuration:**
- [utils/core/config.py](utils/core/config.py) - Dataclass-based config with `load_config()`, supports dict-style access for backward compatibility
- [utils/core/constants.py](utils/core/constants.py) - Enums (`OutputFormat`, `AnnotationType`, `NoiseType`), Taiwan county mappings (`COUNTY_TO_CLASS`), validation constants
- [utils/config.py](utils/config.py) - County mappings, YAML loading, category creation

**Processing Pipeline:**
- [utils/pipeline.py](utils/pipeline.py) - Main processing logic: `process_separate_districts()`, `process_combined_maps()`, `process_single_district_image()`
- [utils/geometry.py](utils/geometry.py) - GIS operations: `load_data()`, `extract_single_district_image()`, `crop_image_and_shapefile()`, window generation
- [utils/image.py](utils/image.py) - Image transforms: noise (`apply_distance_based_noise()`), hue augmentation, rotation, grayscale conversion

**Annotation Writers:**
- [utils/annotations.py](utils/annotations.py) - COCO JSON: `create_annotations_for_image()`, batch buffering, temp-file mode for fast writes
- [utils/writers/yolo.py](utils/writers/yolo.py) - YOLO txt: `write_yolo_annotation_file()`, coordinate normalization, train/val/test split manager

**Mask System:**
- [utils/masks.py](utils/masks.py) - Mask region database for exclusion areas

**Visualization:**
- [utils/visualization.py](utils/visualization.py) - Debug overlays and verification utilities

### Tools (`tools/`)
Standalone utility scripts:
- [tools/clean_mask.py](tools/clean_mask.py) - Interactive mask region selector for excluding zoom-ins, legends
- [tools/convert_yolo_to_coco.py](tools/convert_yolo_to_coco.py) - Convert YOLO segmentation datasets to COCO format

### Processing Modes
- **Separate Districts**: Each image contains ONE county, cropped from full map
- **Combined Maps**: Each image contains ALL counties (full map view)

### Output Formats
- **COCO**: Standard JSON with segmentation polygons or bboxes
- **YOLO**: Per-image txt files with normalized coordinates

### Dataset Split Styles
- **COCO (Coconuts-1)**: `train/`, `valid/`, `test/` with `_annotations.coco.json` in each
- **YOLO**: `train/images/`, `train/labels/`, etc.

## Configuration (YAML)

Key config sections:
- `output.format`: `coco` or `yolo`
- `output.annotation_type`: `segmentation` or `bbox`
- `output.use_split`: Enable train/val/test splitting
- `processing_modes.separate_districts` / `combined_maps`: Enable/disable modes
- `noise_configs`: List of noise augmentation profiles
- `hue_augmentation`: HSV color augmentation
- `rotation`: Random rotation augmentation
- `window_configs`: Sliding window crops (combined mode only)

## Key Patterns

### Coordinate Systems
- GIS coordinates → pixel coordinates via `rasterio.Affine` transform
- YOLO coordinates are normalized [0,1] relative to image dimensions
- All polygon clipping uses Shapely operations

### Batch Processing
- Annotations buffered in memory, flushed periodically (configurable `batch_size`)
- COCO "fast mode" writes individual temp JSON files, merges atomically at end
- Prevents corruption on Ctrl+C interrupts

### Taiwan County IDs
22 counties mapped in `COUNTY_TO_CLASS` (0-21), supporting both Chinese and English names with alternative spellings (臺/台).
