"""
Constants and enumerations for the GIS dataset generator.

This module contains all magic strings, numbers, and enums used throughout
the application to ensure consistency and type safety.
"""

from enum import Enum
from typing import Final


# ============================================================================
# Enumerations
# ============================================================================

class OutputFormat(str, Enum):
    """Supported output annotation formats."""
    COCO = "coco"
    YOLO = "yolo"


class AnnotationType(str, Enum):
    """Types of annotations to generate."""
    SEGMENTATION = "segmentation"
    BBOX = "bbox"


class NoiseType(str, Enum):
    """Types of noise that can be applied to images."""
    GAUSSIAN = "gaussian"
    SALT_PEPPER = "salt_pepper"
    SPECKLE = "speckle"
    PERLIN = "perlin"
    TEXTURED = "textured"


class InterpolationMethod(str, Enum):
    """Image interpolation methods."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"


class ProcessingMode(str, Enum):
    """Dataset processing modes."""
    SEPARATE_DISTRICTS = "separate_districts"
    COMBINED_MAPS = "combined_maps"


# ============================================================================
# File and Path Constants
# ============================================================================

DEFAULT_CONFIG_PATH: Final[str] = "configs/full_coco_config_v2.yaml"
DEFAULT_OUTPUT_DIR: Final[str] = "output"
TEMP_ANNOTATIONS_FILE: Final[str] = "annotations_temp.json"
CONFIG_SNAPSHOT_FILE: Final[str] = "config_used.yaml"

# Directory names
IMAGES_DIR: Final[str] = "images"
LABELS_DIR: Final[str] = "labels"
ANNOTATIONS_DIR: Final[str] = "annotations"
VISUALIZATIONS_DIR: Final[str] = "visualizations"

# File extensions
SUPPORTED_IMAGE_FORMATS: Final[tuple] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
TIF_EXTENSION: Final[str] = '.tif'
SHAPEFILE_EXTENSION: Final[str] = '.shp'


# ============================================================================
# Processing Constants
# ============================================================================

DEFAULT_BATCH_SIZE: Final[int] = 10
DEFAULT_MAX_IMAGE_SIZE: Final[int] = 1024
MIN_POLYGON_AREA: Final[float] = 10.0
MIN_POLYGON_POINTS: Final[int] = 3
MIN_SEGMENTATION_LENGTH: Final[int] = 6  # At least 3 points (x,y pairs)

# Image processing
DEFAULT_CROP_FACTOR: Final[float] = 0.05
DEFAULT_FILL_VALUE: Final[int] = 0
MAX_HUE_SHIFT: Final[float] = 1.0
MIN_SATURATION: Final[float] = 0.0
MAX_SATURATION: Final[float] = 2.0

# Noise parameters
DEFAULT_NOISE_INTENSITY: Final[float] = 0.5
DEFAULT_BORDER_BUFFER_PIXELS: Final[int] = 8
NOISE_ACCELERATION_EPSILON: Final[float] = 1e-6

# Coordinate normalization
NORMALIZED_MIN: Final[float] = 0.0
NORMALIZED_MAX: Final[float] = 1.0


# ============================================================================
# Taiwan County Mappings
# ============================================================================

COUNTY_ENGLISH_NAMES: Final[dict] = {
    '臺北市': 'Taipei_City',
    '新北市': 'New_Taipei_City',
    '基隆市': 'Keelung_City',
    '桃園市': 'Taoyuan_City',
    '新竹市': 'Hsinchu_City',
    '新竹縣': 'Hsinchu_County',
    '苗栗縣': 'Miaoli_County',
    '臺中市': 'Taichung_City',
    '彰化縣': 'Changhua_County',
    '南投縣': 'Nantou_County',
    '雲林縣': 'Yunlin_County',
    '嘉義市': 'Chiayi_City',
    '嘉義縣': 'Chiayi_County',
    '臺南市': 'Tainan_City',
    '高雄市': 'Kaohsiung_City',
    '屏東縣': 'Pingtung_County',
    '宜蘭縣': 'Yilan_County',
    '花蓮縣': 'Hualien_County',
    '臺東縣': 'Taitung_County',
    '澎湖縣': 'Penghu_County',
    '金門縣': 'Kinmen_County',
    '連江縣': 'Lienchiang_County',
}

# Alternative spellings
COUNTY_ENGLISH_NAMES_ALT: Final[dict] = {
    '台北市': 'Taipei_City',
    '台中市': 'Taichung_City',
    '台南市': 'Tainan_City',
    '台東縣': 'Taitung_County',
}

# Merge both mappings
ALL_COUNTY_NAMES = {**COUNTY_ENGLISH_NAMES, **COUNTY_ENGLISH_NAMES_ALT}

# COCO category IDs (0-indexed)
COUNTY_TO_CLASS: Final[dict] = {
    'Taipei_City': 0,
    'New_Taipei_City': 1,
    'Keelung_City': 2,
    'Taoyuan_City': 3,
    'Hsinchu_City': 4,
    'Hsinchu_County': 5,
    'Miaoli_County': 6,
    'Taichung_City': 7,
    'Changhua_County': 8,
    'Nantou_County': 9,
    'Yunlin_County': 10,
    'Chiayi_City': 11,
    'Chiayi_County': 12,
    'Tainan_City': 13,
    'Kaohsiung_City': 14,
    'Pingtung_County': 15,
    'Yilan_County': 16,
    'Hualien_County': 17,
    'Taitung_County': 18,
    'Penghu_County': 19,
    'Kinmen_County': 20,
    'Lienchiang_County': 21,
}

# Northern Taiwan districts (for special handling)
NORTHERN_DISTRICTS: Final[frozenset] = frozenset([
    'taipei', 'new_taipei', 'keelung', 'taoyuan', 'hsinchu', 'yilan'
])


# ============================================================================
# Shapefile Column Names
# ============================================================================

SHAPEFILE_COUNTY_COLUMNS: Final[tuple] = (
    'COUNTYENG', 'COUNTYNAME', 'NAME', 'County', 'county',
    'COUNTY', 'C_Name', '縣市名稱'
)


# ============================================================================
# Validation Messages
# ============================================================================

class ValidationMessages:
    """Standard validation error messages."""
    INVALID_FORMAT = "Invalid output format: {format}. Must be one of: {options}"
    INVALID_ANNOTATION_TYPE = "Invalid annotation type: {type}. Must be one of: {options}"
    INVALID_NOISE_TYPE = "Invalid noise type: {type}. Must be one of: {options}"
    INVALID_INTERPOLATION = "Invalid interpolation method: {method}. Must be one of: {options}"
    FILE_NOT_FOUND = "File not found: {path}"
    DIRECTORY_NOT_FOUND = "Directory not found: {path}"
    INVALID_BATCH_SIZE = "Batch size must be positive, got: {size}"
    INVALID_IMAGE_SIZE = "Image size must be positive, got: {size}"
    EMPTY_DISTRICT_LIST = "Districts list cannot be empty"
    MISSING_REQUIRED_KEY = "Required configuration key missing: {key}"


# ============================================================================
# Logging Configuration
# ============================================================================

LOG_FORMAT: Final[str] = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT: Final[str] = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_LEVEL: Final[str] = 'INFO'
