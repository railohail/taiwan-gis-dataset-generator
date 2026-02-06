#!/usr/bin/env python3
"""
Hierarchical Dataset Generator for H-DETR

Generates dual-level YOLO annotations:
- L0: County (縣市) - 22 classes
- L1: Township (鄉鎮區) - 368 classes

Usage:
    python generate_hierarchical.py
    python generate_hierarchical.py --config configs/hierarchical-yolo-seg.yaml
    python generate_hierarchical.py --verbose
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.core import load_config, setup_logger
from utils.hierarchical_pipeline import run_hierarchical_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate hierarchical dataset for H-DETR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_hierarchical.py
  python generate_hierarchical.py --config configs/hierarchical-yolo-seg.yaml
  python generate_hierarchical.py --verbose
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=Path('configs/hierarchical-yolo-seg.yaml'),
        help='Path to hierarchical config file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(name="HierarchicalGenerator", level=log_level, console=True)

    logger.info("=" * 70)
    logger.info("HIERARCHICAL DATASET GENERATOR FOR H-DETR")
    logger.info("=" * 70)

    # Check config exists
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info("Create a hierarchical config or use: configs/hierarchical-yolo-seg.yaml")
        return 1

    logger.info(f"Config: {args.config}")

    # Load config
    config = load_config(str(args.config))

    # Verify hierarchical mode is enabled
    if not config.get('hierarchy', {}).get('enabled', False):
        logger.error("Hierarchical mode not enabled in config!")
        logger.info("Set 'hierarchy.enabled: true' in your config file")
        return 1

    # Run pipeline
    try:
        images, l0_anns, l1_anns = run_hierarchical_pipeline(config=config)

        logger.info("=" * 70)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total images: {images}")
        logger.info(f"L0 (County) annotations: {l0_anns}")
        logger.info(f"L1 (Township) annotations: {l1_anns}")
        logger.info(f"Output: {config.get('output_base_dir', 'yolo_hierarchical')}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
