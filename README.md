# Taiwan GIS Dataset Generator

A production-ready tool for converting Taiwan county GIS map data (TIF raster files with shapefile annotations) into machine learning datasets in COCO or YOLO format. Supports instance segmentation and object detection tasks.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Main Generator](#main-generator)
  - [Command Line Arguments](#command-line-arguments)
- [Configuration](#configuration)
  - [Basic Settings](#basic-settings)
  - [Output Settings](#output-settings)
  - [Processing Modes](#processing-modes)
  - [Augmentations](#augmentations)
  - [Noise Configuration](#noise-configuration)
  - [Window Configuration](#window-configuration)
  - [Performance Settings](#performance-settings)
- [Tools](#tools)
  - [Interactive Mask Tool](#interactive-mask-tool)
  - [YOLO to COCO Converter](#yolo-to-coco-converter)
- [Output Formats](#output-formats)
- [Directory Structure](#directory-structure)
- [Taiwan County Classes](#taiwan-county-classes)

## Requirements

- Python 3.8+
- Dependencies:
  - rasterio
  - geopandas
  - shapely
  - opencv-python
  - numpy
  - PyYAML
  - Pillow
  - tqdm
  - matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/taiwan-gis-dataset-generator.git
cd taiwan-gis-dataset-generator

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Place your TIF files in `datasets/MAPDATA/<district_name>/`
2. Place your shapefile in `datasets/shapefile/`
3. Run the generator:

```bash
python main.py --config configs/yolo-seg.yaml
```

## Usage

### Main Generator

The main entry point is `main.py`, which orchestrates the entire dataset generation pipeline.

```bash
# Use default configuration
python main.py

# Use custom configuration file
python main.py --config path/to/config.yaml

# Enable verbose logging
python main.py --verbose

# Combine options
python main.py --config configs/yolo-seg.yaml --verbose
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--config` | `-c` | Path to YAML configuration file |
| `--verbose` | `-v` | Enable DEBUG level logging |
| `--version` | | Show version information |
| `--help` | `-h` | Show help message |

## Configuration

Configuration is done via YAML files. Below are all available options with descriptions.

### Basic Settings

```yaml
# Path to county shapefile
shapefile_path: datasets/shapefile/COUNTY_MOI_1130718.shp

# Base directory containing TIF files organized by district
mapdata_base_dir: datasets/MAPDATA

# Output directory for generated dataset
output_base_dir: yolo_seg_output

# Crop factor for combined maps (removes border artifacts)
# 0.05 = crop 5% from each edge
crop_factor: 0.05

# List of districts to process (folder names in mapdata_base_dir)
districts:
  - miaoli
  - taipei
  - new_taipei
  - keelung
  - taoyuan
  - hsinchu
  - yilan
```

### Output Settings

```yaml
output:
  # Output format: 'coco' or 'yolo'
  format: yolo

  # Annotation type: 'segmentation' (polygon masks) or 'bbox' (bounding boxes)
  annotation_type: segmentation

  # Convert images to grayscale
  grayscale: false

  # Enable train/val/test splits
  use_split: true

  # Split ratios (must sum to 1.0)
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1

  # Random seed for reproducible splits
  split_seed: 42
```

### Processing Modes

```yaml
processing_modes:
  # Separate Districts: Each image contains ONE county, cropped from full map
  separate_districts: true

  # Combined Maps: Each image contains ALL counties (full map view)
  combined_maps: true
```

### Processing Parameters

```yaml
processing:
  # Maximum TIF files to process per district (null = no limit)
  max_files_per_district: 10

  # Minimum polygon area in pixels (filters tiny artifacts)
  min_polygon_area: 100

  # Skip already processed files
  skip_existing: false

  # Skip separate district images if mask coverage exceeds this percentage
  # Example: 50.0 means skip if >50% is masked
  mask_skip_threshold_separate: 50.0
```

### Augmentations

#### Hue Augmentation

```yaml
hue_augmentation:
  # Enable/disable hue augmentation
  enabled: true

  # Number of color-shifted variants per image
  count: 3

  # Hue shift range (-1.0 to 1.0)
  hue_shift_range: [-0.3, 0.3]

  # Saturation multiplier range
  saturation_range: [0.8, 1.2]

  # Value/brightness multiplier range
  value_range: [0.9, 1.1]
```

#### Rotation Augmentation

```yaml
rotation:
  # Enable/disable rotation
  enabled: true

  # Number of rotated variants per image
  count: 2

  # Rotation angle range in degrees
  angle_range: [-30, 30]

  # Fill value for empty regions after rotation (0 = black)
  fill_value: 0

  # Interpolation method: 'nearest', 'bilinear', or 'cubic'
  interpolation: bilinear
```

### Noise Configuration

Apply distance-based noise that increases toward polygon boundaries:

```yaml
noise_configs:
  # Clean (no noise) - always include for original images
  - name: clean
    enabled: true
    type: gaussian
    intensity: 0.0
    acceleration: 0
    border_buffer_pixels: 0

  # Light noise
  - name: light
    enabled: true
    type: gaussian
    intensity: 0.4
    acceleration: 1.0
    border_buffer_pixels: 10

  # Medium noise
  - name: medium
    enabled: true
    type: gaussian
    intensity: 0.6
    acceleration: 0.8
    border_buffer_pixels: 8

  # Heavy noise
  - name: hard
    enabled: true
    type: gaussian
    intensity: 0.9
    acceleration: 0.6
    border_buffer_pixels: 8
```

Noise parameters:
- `type`: Noise type (`gaussian`, `salt_pepper`, `speckle`, `perlin`, `textured`)
- `intensity`: Noise strength (0.0 to 1.0)
- `acceleration`: How quickly noise increases near boundaries
- `border_buffer_pixels`: Pixel buffer around polygon edges

### Window Configuration

Sliding window crops for combined maps mode:

```yaml
window_configs:
  # Full image (100% x 100%)
  - name: xy_100
    x_percent: 100
    y_percent: 100

  # 80% crop
  - name: xy_80
    x_percent: 80
    y_percent: 80

  # Asymmetric crop
  - name: x_80_y_50
    x_percent: 80
    y_percent: 50
```

### Performance Settings

```yaml
performance:
  # Flush annotations to disk every N images
  # Lower = safer (more frequent saves), higher = faster
  # Set to 1 when using splits for correct label placement
  batch_size: 10
```

### Visualization Settings

```yaml
visualization:
  # Create debug mask images
  create_masks: true

  # DPI for saved visualizations
  dpi: 300

  # Save noise debug images
  save_noise_debug: true
```

## Tools

### Interactive Mask Tool

Define exclusion regions (zoom-ins, legends, decorative elements) that should not be included in annotations.

```bash
python -m tools.clean_mask
```

Controls:
- Click & Drag: Draw mask rectangle
- `r`: Reset/clear all rectangles
- `n`: Save and generate visualization
- `s`: Skip current image
- `q`: Quit

Output:
- `mask_database/masks.yaml`: Persistent mask database
- `interact/visualizations/`: Before/after comparison images

### YOLO to COCO Converter

Convert YOLO segmentation datasets to COCO format:

```bash
# Default directories
python -m tools.convert_yolo_to_coco

# Custom directories
python -m tools.convert_yolo_to_coco --input yolo_dataset --output coco_dataset
```

Arguments:
- `--input`: Input YOLO dataset directory (default: `yolo_seg_base`)
- `--output`: Output COCO dataset directory (default: `coco_seg_base`)

## Output Formats

### YOLO Format

```
output_dir/
  classes.txt           # Class names (one per line)
  dataset.yaml          # Dataset configuration for training
  train/
    images/
      image1.png
    labels/
      image1.txt        # Normalized polygon coordinates
  val/
    images/
    labels/
  test/
    images/
    labels/
```

YOLO label format (segmentation):
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
All coordinates are normalized to [0, 1].

### COCO Format (Coconuts-1 Style)

```
output_dir/
  train/
    _annotations.coco.json
    image1.png
    image2.png
  valid/
    _annotations.coco.json
    image1.png
  test/
    _annotations.coco.json
    image1.png
```

## Directory Structure

### Input Data Structure

```
datasets/
  MAPDATA/
    miaoli/
      map1.tif
      map2.tif
    taipei/
      map1.tif
  shapefile/
    COUNTY_MOI_1130718.shp
    COUNTY_MOI_1130718.shx
    COUNTY_MOI_1130718.dbf
    COUNTY_MOI_1130718.prj
```

### Project Structure

```
taiwan-gis-dataset-generator/
  main.py                 # Main entry point
  configs/
    yolo-seg.yaml         # Example YOLO segmentation config
  tools/
    clean_mask.py         # Interactive mask tool
    convert_yolo_to_coco.py  # Format converter
  utils/
    config.py             # Configuration loading
    annotations.py        # COCO annotation handling
    geometry.py           # GIS operations
    image.py              # Image processing
    pipeline.py           # Processing workflow
    masks.py              # Mask database
    visualization.py      # Debug visualizations
    writers/
      yolo.py             # YOLO format writer
    core/
      config.py           # Dataclass configuration
      constants.py        # Enums and constants
      logger.py           # Logging utilities
  datasets/               # Input data (not tracked in git)
  mask_database/          # Persistent mask storage
```

## Taiwan County Classes

The generator supports 22 Taiwan counties with the following class IDs:

| ID | English Name | Chinese Name |
|----|--------------|--------------|
| 0 | Changhua County | 彰化縣 |
| 1 | Chiayi City | 嘉義市 |
| 2 | Chiayi County | 嘉義縣 |
| 3 | Hsinchu City | 新竹市 |
| 4 | Hsinchu County | 新竹縣 |
| 5 | Hualien County | 花蓮縣 |
| 6 | Kaohsiung City | 高雄市 |
| 7 | Keelung City | 基隆市 |
| 8 | Kinmen County | 金門縣 |
| 9 | Lienchiang County | 連江縣 |
| 10 | Miaoli County | 苗栗縣 |
| 11 | Nantou County | 南投縣 |
| 12 | New Taipei City | 新北市 |
| 13 | Penghu County | 澎湖縣 |
| 14 | Pingtung County | 屏東縣 |
| 15 | Taichung City | 台中市 |
| 16 | Tainan City | 台南市 |
| 17 | Taipei City | 台北市 |
| 18 | Taitung County | 台東縣 |
| 19 | Taoyuan City | 桃園市 |
| 20 | Yilan County | 宜蘭縣 |
| 21 | Yunlin County | 雲林縣 |

## Example Configuration

A minimal configuration for YOLO segmentation output:

```yaml
# Paths
shapefile_path: datasets/shapefile/COUNTY_MOI_1130718.shp
mapdata_base_dir: datasets/MAPDATA
output_base_dir: yolo_output

# Districts to process
districts:
  - miaoli
  - taipei

# Output settings
output:
  format: yolo
  annotation_type: segmentation
  use_split: true
  train_ratio: 0.8
  val_ratio: 0.2
  test_ratio: 0.0

# Processing modes
processing_modes:
  separate_districts: true
  combined_maps: false

# Minimal augmentation
noise_configs:
  - name: clean
    enabled: true
    type: gaussian
    intensity: 0.0
    acceleration: 0
    border_buffer_pixels: 0

# No rotation or hue augmentation
rotation:
  enabled: false

hue_augmentation:
  enabled: false

# Performance
performance:
  batch_size: 10

processing:
  min_polygon_area: 100
```

## License

See LICENSE file for details.
