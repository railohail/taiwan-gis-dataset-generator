"""
Configuration management for GIS dataset generation.

This module provides:
- Taiwan county name mappings (Chinese/English)
- Category ID mappings for ML datasets
- YAML configuration loading
- Default configuration creation

Note: This is the primary configuration module. The canonical COUNTY_TO_CLASS
mapping lives in core/constants.py; this module extends it with Chinese names.
"""

import yaml
import os

# --- Configuration Parameters ---
CONFIG_PATH = 'configs/full_coco_config_v2.yaml'
OUTPUT_BASE_DIR = 'full_coco_output_v2'

# English translation mapping for Taiwan counties/cities
COUNTY_ENGLISH_NAMES = {
    '連江縣': 'Lienchiang County',
    '宜蘭縣': 'Yilan County',
    '彰化縣': 'Changhua County',
    '南投縣': 'Nantou County',
    '雲林縣': 'Yunlin County',
    '屏東縣': 'Pingtung County',
    '基隆市': 'Keelung City',
    '臺北市': 'Taipei City',
    '新北市': 'New Taipei City',
    '臺中市': 'Taichung City',
    '臺南市': 'Tainan City',
    '桃園市': 'Taoyuan City',
    '苗栗縣': 'Miaoli County',
    '嘉義市': 'Chiayi City',
    '嘉義縣': 'Chiayi County',
    '金門縣': 'Kinmen County',
    '高雄市': 'Kaohsiung City',
    '臺東縣': 'Taitung County',
    '花蓮縣': 'Hualien County',
    '澎湖縣': 'Penghu County',
    '新竹市': 'Hsinchu City',
    '新竹縣': 'Hsinchu County'
}

# Category mapping - CANONICAL SOURCE
# Use the core/constants.py COUNTY_TO_CLASS as the single source of truth
# This module provides Chinese name lookups that delegate to the canonical English mapping
from .core.constants import COUNTY_TO_CLASS as _ENGLISH_COUNTY_TO_CLASS

# Chinese to English name mapping for category lookup
_CHINESE_TO_ENGLISH = {
    '連江縣': 'Lienchiang_County',
    '宜蘭縣': 'Yilan_County',
    '彰化縣': 'Changhua_County',
    '南投縣': 'Nantou_County',
    '雲林縣': 'Yunlin_County',
    '屏東縣': 'Pingtung_County',
    '基隆市': 'Keelung_City',
    '臺北市': 'Taipei_City',
    '台北市': 'Taipei_City',  # Alternative spelling
    '新北市': 'New_Taipei_City',
    '臺中市': 'Taichung_City',
    '台中市': 'Taichung_City',  # Alternative spelling
    '臺南市': 'Tainan_City',
    '台南市': 'Tainan_City',  # Alternative spelling
    '桃園市': 'Taoyuan_City',
    '苗栗縣': 'Miaoli_County',
    '嘉義市': 'Chiayi_City',
    '嘉義縣': 'Chiayi_County',
    '金門縣': 'Kinmen_County',
    '高雄市': 'Kaohsiung_City',
    '臺東縣': 'Taitung_County',
    '台東縣': 'Taitung_County',  # Alternative spelling
    '花蓮縣': 'Hualien_County',
    '澎湖縣': 'Penghu_County',
    '新竹市': 'Hsinchu_City',
    '新竹縣': 'Hsinchu_County',
}

# Build COUNTY_TO_CLASS that supports BOTH Chinese and English lookups
# This ensures backward compatibility while using canonical IDs from core/constants.py
COUNTY_TO_CLASS = {}
# Add English names (canonical)
COUNTY_TO_CLASS.update(_ENGLISH_COUNTY_TO_CLASS)
# Add Chinese names (mapped to same IDs)
for chinese_name, english_name in _CHINESE_TO_ENGLISH.items():
    if english_name in _ENGLISH_COUNTY_TO_CLASS:
        COUNTY_TO_CLASS[chinese_name] = _ENGLISH_COUNTY_TO_CLASS[english_name]

# Reverse mapping: English → Chinese (for shapefile compatibility)
ENGLISH_TO_COUNTY = {v: k for k, v in COUNTY_ENGLISH_NAMES.items()}


def normalize_county_name(county_name):
    """
    Normalize county name to Chinese format for COUNTY_TO_CLASS lookup.
    Supports both Chinese and English input names.

    Args:
        county_name: County name in Chinese or English

    Returns:
        Chinese county name if found, original name otherwise

    Example:
        normalize_county_name("Taipei City") -> "臺北市"
        normalize_county_name("臺北市") -> "臺北市"
    """
    # Already in Chinese format
    if county_name in COUNTY_TO_CLASS:
        return county_name

    # Convert from English to Chinese
    if county_name in ENGLISH_TO_COUNTY:
        return ENGLISH_TO_COUNTY[county_name]

    # Unknown county name
    return county_name


def create_default_config():
    """Create a default config file if it doesn't exist."""
    default_config = {
        'districts': ['kaoshung'],  # List of district folders to process
        'shapefile_path': 'datasets/shapefile/COUNTY_MOI_1130718.shp',
        'mapdata_base_dir': 'datasets/MAPDATA',
        'output_base_dir': 'full_coco_output_v2',
        'crop_factor': 0.05,

        # Processing modes for v2
        'processing_modes': {
            'separate_districts': True,   # Each district gets its own image
            'combined_maps': True         # All districts combined in one image
        },

        # Multiple window configurations
        'window_configs': [
            {'name': 'small', 'x_percent': 40, 'y_percent': 40},
            {'name': 'medium', 'x_percent': 50, 'y_percent': 50},
            {'name': 'large', 'x_percent': 60, 'y_percent': 60}
        ],

        # Multiple noise configurations
        'noise_configs': [
            {
                'name': 'light',
                'enabled': True,
                'intensity': 0.4,
                'type': 'gaussian',
                'border_buffer_pixels': 10,
                'acceleration': 1.0
            },
            {
                'name': 'medium',
                'enabled': True,
                'intensity': 0.6,
                'type': 'gaussian',
                'border_buffer_pixels': 8,
                'acceleration': 1.2
            }
        ],

        # Hue augmentation configuration (NEW for v2)
        'hue_augmentation': {
            'enabled': True,
            'count': 3,  # Number of hue-shifted versions
            'hue_shift_range': [-0.3, 0.3],  # Range of hue shifts (-1 to 1)
            'saturation_range': [0.8, 1.2],  # Range of saturation multipliers
            'value_range': [0.9, 1.1]  # Range of value/brightness multipliers
        },

        # Rotation configuration
        'rotation': {
            'enabled': True,
            'count': 2,
            'angle_range': [-30, 30],
            'interpolation': 'bilinear',
            'fill_value': 0
        },

        # Visualization settings
        'visualization': {
            'create_masks': True,
            'dpi': 300,
            'save_noise_debug': False,  # Disabled for performance
            'create_debug_overlays': False  # NEW: Disable debug overlays for speed
        },

        # Processing settings
        'processing': {
            'max_files_per_district': 2,  # Limit files for testing
            'skip_existing': False,
            'min_polygon_area': 10,
            'mask_skip_threshold_separate': 50.0  # Skip separate districts if >50% masked
        },

        # Performance optimization settings (NEW)
        'performance': {
            'batch_size': 100,  # Number of images to buffer before writing to JSON
            'verify_images': False,  # Skip reading images after saving (faster)
            'verbose_verification': False  # Reduce console output during verification
        }
    }

    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    print(f"Created default config file: {CONFIG_PATH}")
    return default_config


def load_config():
    """Load configuration from YAML file."""
    if not os.path.exists(CONFIG_PATH):
        return create_default_config()

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {CONFIG_PATH}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return create_default_config()


def create_categories():
    """Create COCO categories from county mapping."""
    categories = []
    for county_name, class_id in COUNTY_TO_CLASS.items():
        english_name = COUNTY_ENGLISH_NAMES.get(county_name, county_name)
        categories.append({
            'id': class_id,
            'name': english_name,
            'supercategory': 'region'
        })
    return categories
