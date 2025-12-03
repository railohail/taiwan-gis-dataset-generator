"""
Configuration management with validation.

Loads and validates YAML configuration files, providing type-safe
access to configuration parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml

from .constants import (
    OutputFormat,
    AnnotationType,
    NoiseType,
    InterpolationMethod,
    ProcessingMode,
    DEFAULT_CONFIG_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_CROP_FACTOR,
    ValidationMessages,
    COUNTY_TO_CLASS,
)
from .logger import get_logger

logger = get_logger(__name__)


class DictAccessMixin:
    """
    Mixin to add dictionary-style access to dataclasses.

    This enables backward compatibility with legacy code that expects
    dictionary-style access (config['key'] or config.get('key', default)).
    """

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access config['key']."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"No attribute: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Support dictionary-style get() method."""
        return getattr(self, key, default)


@dataclass
class OutputConfig(DictAccessMixin):
    """Output format configuration."""
    format: OutputFormat = OutputFormat.COCO
    annotation_type: AnnotationType = AnnotationType.SEGMENTATION
    grayscale: bool = False
    # Dataset split configuration
    use_split: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    split_seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'OutputConfig':
        """Create from dictionary."""
        return cls(
            format=OutputFormat(data.get('format', 'coco')),
            annotation_type=AnnotationType(data.get('annotation_type', 'segmentation')),
            grayscale=data.get('grayscale', False),
            use_split=data.get('use_split', True),
            train_ratio=data.get('train_ratio', 0.7),
            val_ratio=data.get('val_ratio', 0.2),
            test_ratio=data.get('test_ratio', 0.1),
            split_seed=data.get('split_seed', 42)
        )


@dataclass
class NoiseConfig(DictAccessMixin):
    """Noise augmentation configuration."""
    name: str
    enabled: bool
    intensity: float
    type: NoiseType
    border_buffer_pixels: int
    acceleration: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'NoiseConfig':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            enabled=data.get('enabled', True),
            intensity=data.get('intensity', 0.5),
            type=NoiseType(data.get('type', 'gaussian')),
            border_buffer_pixels=data.get('border_buffer_pixels', 8),
            acceleration=data.get('acceleration', 1.0)
        )


@dataclass
class HueAugmentationConfig(DictAccessMixin):
    """Hue augmentation configuration."""
    enabled: bool = False
    count: int = 3
    hue_shift_range: tuple[float, float] = (-0.3, 0.3)
    saturation_range: tuple[float, float] = (0.8, 1.2)
    value_range: tuple[float, float] = (0.9, 1.1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'HueAugmentationConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', False),
            count=data.get('count', 3),
            hue_shift_range=tuple(data.get('hue_shift_range', [-0.3, 0.3])),
            saturation_range=tuple(data.get('saturation_range', [0.8, 1.2])),
            value_range=tuple(data.get('value_range', [0.9, 1.1]))
        )


@dataclass
class RotationConfig(DictAccessMixin):
    """Rotation augmentation configuration."""
    enabled: bool = False
    count: int = 2
    angle_range: tuple[float, float] = (-30, 30)
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
    fill_value: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RotationConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', False),
            count=data.get('count', 2),
            angle_range=tuple(data.get('angle_range', [-30, 30])),
            interpolation=InterpolationMethod(data.get('interpolation', 'bilinear')),
            fill_value=data.get('fill_value', 0)
        )


@dataclass
class WindowConfig(DictAccessMixin):
    """Sliding window configuration."""
    name: str
    x_percent: float
    y_percent: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WindowConfig':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            x_percent=data['x_percent'],
            y_percent=data['y_percent']
        )


@dataclass
class ProcessingModesConfig(DictAccessMixin):
    """Processing modes configuration."""
    separate_districts: bool = True
    combined_maps: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProcessingModesConfig':
        """Create from dictionary."""
        return cls(
            separate_districts=data.get('separate_districts', True),
            combined_maps=data.get('combined_maps', True)
        )


@dataclass
class ProcessingConfig(DictAccessMixin):
    """General processing configuration."""
    max_files_per_district: Optional[int] = None
    min_polygon_area: float = 10.0
    skip_existing: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProcessingConfig':
        """Create from dictionary."""
        return cls(
            max_files_per_district=data.get('max_files_per_district'),
            min_polygon_area=data.get('min_polygon_area', 10.0),
            skip_existing=data.get('skip_existing', False)
        )


@dataclass
class PerformanceConfig(DictAccessMixin):
    """Performance configuration."""
    batch_size: int = DEFAULT_BATCH_SIZE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PerformanceConfig':
        """Create from dictionary."""
        batch_size = data.get('batch_size', DEFAULT_BATCH_SIZE)
        if batch_size <= 0:
            raise ValueError(ValidationMessages.INVALID_BATCH_SIZE.format(size=batch_size))
        return cls(batch_size=batch_size)


@dataclass
class VisualizationConfig(DictAccessMixin):
    """Visualization configuration."""
    create_masks: bool = True
    dpi: int = 300
    save_noise_debug: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'VisualizationConfig':
        """Create from dictionary."""
        return cls(
            create_masks=data.get('create_masks', True),
            dpi=data.get('dpi', 300),
            save_noise_debug=data.get('save_noise_debug', False)
        )


@dataclass
class Config:
    """Main configuration class."""
    # Required fields
    output_base_dir: str
    districts: list[str]
    shapefile_path: str
    mapdata_base_dir: str

    # Optional fields with defaults
    crop_factor: float = DEFAULT_CROP_FACTOR
    output: OutputConfig = field(default_factory=OutputConfig)
    processing_modes: ProcessingModesConfig = field(default_factory=ProcessingModesConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    noise_configs: list[NoiseConfig] = field(default_factory=list)
    hue_augmentation: HueAugmentationConfig = field(default_factory=HueAugmentationConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    window_configs: list[WindowConfig] = field(default_factory=list)

    def validate(self) -> None:
        """Validate configuration."""
        # Check required fields
        if not self.districts:
            raise ValueError(ValidationMessages.EMPTY_DISTRICT_LIST)

        # Check paths exist
        shapefile = Path(self.shapefile_path)
        if not shapefile.exists():
            raise FileNotFoundError(
                ValidationMessages.FILE_NOT_FOUND.format(path=self.shapefile_path)
            )

        mapdata_dir = Path(self.mapdata_base_dir)
        if not mapdata_dir.exists():
            raise FileNotFoundError(
                ValidationMessages.DIRECTORY_NOT_FOUND.format(path=self.mapdata_base_dir)
            )

        # Validate ranges
        if self.crop_factor < 0 or self.crop_factor > 1:
            raise ValueError(f"crop_factor must be between 0 and 1, got {self.crop_factor}")

        logger.info("Configuration validated successfully")

    def __getitem__(self, key: str) -> Any:
        """
        Support dictionary-style access for backward compatibility.

        This allows legacy code to use config['key'] syntax.
        """
        if hasattr(self, key):
            value = getattr(self, key)
            # Wrap nested dataclass configs to support dict-style access
            if isinstance(value, (OutputConfig, ProcessingModesConfig, ProcessingConfig,
                                PerformanceConfig, VisualizationConfig, HueAugmentationConfig,
                                RotationConfig)):
                return DataclassDict(value)
            return value
        raise KeyError(f"Configuration has no key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Support dictionary-style get() for backward compatibility.

        This allows legacy code to use config.get('key', default) syntax.
        """
        try:
            return self[key]
        except KeyError:
            return default

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        # Process nested configurations
        output = OutputConfig.from_dict(data.get('output', {}))
        processing_modes = ProcessingModesConfig.from_dict(data.get('processing_modes', {}))
        processing = ProcessingConfig.from_dict(data.get('processing', {}))
        performance = PerformanceConfig.from_dict(data.get('performance', {}))
        visualization = VisualizationConfig.from_dict(data.get('visualization', {}))
        hue_augmentation = HueAugmentationConfig.from_dict(data.get('hue_augmentation', {}))
        rotation = RotationConfig.from_dict(data.get('rotation', {}))

        # Process lists
        noise_configs = [
            NoiseConfig.from_dict(nc) for nc in data.get('noise_configs', [])
        ]
        window_configs = [
            WindowConfig.from_dict(wc) for wc in data.get('window_configs', [])
        ]

        config = cls(
            output_base_dir=data['output_base_dir'],
            districts=data['districts'],
            shapefile_path=data['shapefile_path'],
            mapdata_base_dir=data['mapdata_base_dir'],
            crop_factor=data.get('crop_factor', DEFAULT_CROP_FACTOR),
            output=output,
            processing_modes=processing_modes,
            processing=processing,
            performance=performance,
            visualization=visualization,
            noise_configs=noise_configs,
            hue_augmentation=hue_augmentation,
            rotation=rotation,
            window_configs=window_configs
        )

        config.validate()
        return config


class DataclassDict:
    """
    Wrapper to make dataclass instances behave like dictionaries.

    This enables legacy code to use dict-style access on config objects.
    """

    def __init__(self, dataclass_obj: Any):
        self._obj = dataclass_obj

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute with default value."""
        return getattr(self._obj, key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        if hasattr(self._obj, key):
            return getattr(self._obj, key)
        raise KeyError(f"No attribute: {key}")

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped object."""
        return getattr(self._obj, name)


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            ValidationMessages.FILE_NOT_FOUND.format(path=config_path)
        )

    logger.info(f"Loading configuration from: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    config = Config.from_dict(data)
    logger.info(f"Configuration loaded: {len(config.districts)} districts, "
                f"{len(config.noise_configs)} noise configs")

    return config


def create_categories() -> list[dict[str, Any]]:
    """
    Create COCO categories list.

    Returns:
        List of category dictionaries
    """
    return [
        {"id": class_id, "name": name, "supercategory": "county"}
        for name, class_id in sorted(COUNTY_TO_CLASS.items(), key=lambda x: x[1])
    ]
