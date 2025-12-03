"""
Core utilities for the GIS dataset generator.

This package contains fundamental components like constants, configuration,
and logging that are used throughout the application.
"""

from .constants import (
    OutputFormat,
    AnnotationType,
    NoiseType,
    InterpolationMethod,
    ProcessingMode,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_IMAGE_SIZE,
    MIN_POLYGON_AREA,
    SUPPORTED_IMAGE_FORMATS,
    DEFAULT_CONFIG_PATH,
    COUNTY_TO_CLASS,
    COUNTY_ENGLISH_NAMES,
)
from .config import Config, load_config, create_categories
from .logger import setup_logger, get_logger

__all__ = [
    # Enums
    "OutputFormat",
    "AnnotationType",
    "NoiseType",
    "InterpolationMethod",
    "ProcessingMode",
    # Constants
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_IMAGE_SIZE",
    "MIN_POLYGON_AREA",
    "SUPPORTED_IMAGE_FORMATS",
    "DEFAULT_CONFIG_PATH",
    "COUNTY_TO_CLASS",
    "COUNTY_ENGLISH_NAMES",
    # Config
    "Config",
    "load_config",
    "create_categories",
    # Logging
    "setup_logger",
    "get_logger",
]
