#!/usr/bin/env python3
"""
GIS COCO/YOLO Dataset Generator - Main Entry Point

A production-ready dataset generator for Taiwan county GIS map data.
Supports multiple output formats (COCO, YOLO) and annotation types
(segmentation, bounding box).

Usage:
    python main.py [--config path/to/config.yaml] [--verbose]

Features:
    - COCO and YOLO format output
    - Segmentation and bounding box annotations
    - Multiple augmentations (noise, hue, rotation)
    - Grayscale conversion
    - Batch processing for efficiency
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Core imports
from utils.core import (
    Config,
    load_config,
    setup_logger,
    get_logger,
    OutputFormat,
    AnnotationType,
    DEFAULT_CONFIG_PATH,
)

# I/O imports (will be available after full refactoring)
# from utils.io import COCOWriter, YOLOWriter

# Dataset generation utilities
from utils.config import create_categories
from utils.annotations import initialize_coco_file, set_batch_size
from utils.writers.yolo import initialize_yolo_dataset, set_yolo_batch_size, initialize_split_manager, print_split_statistics
from utils.pipeline import process_separate_districts, process_combined_maps, reset_processing_state
from utils.visualization import verify_all_image_dimensions


class DatasetGenerator:
    """
    Main dataset generator orchestrator.

    Handles the end-to-end process of converting GIS data to
    annotated datasets in COCO or YOLO format.
    """

    def __init__(self, config_path: Optional[Path] = None, verbose: bool = False):
        """
        Initialize the dataset generator.

        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
        """
        self.config_path = config_path or Path(DEFAULT_CONFIG_PATH)
        self.verbose = verbose

        # Setup logging
        log_level = "DEBUG" if verbose else "INFO"
        self.logger = setup_logger(
            name="DatasetGenerator",
            level=log_level,
            console=True
        )

        # Load configuration
        self.logger.info(f"Loading configuration from: {self.config_path}")
        self.config = load_config(str(self.config_path))

        # Statistics
        self.total_images = 0
        self.total_annotations = 0

    def setup_output_directories(self) -> tuple[Path, Optional[Path]]:
        """
        Create output directory structure.

        Returns:
            Tuple of (coco_file_path, labels_dir)
        """
        base_output_dir = Path(self.config['output_base_dir'])
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Output directory: {base_output_dir}")

        # Save configuration snapshot
        config_snapshot = base_output_dir / "config_used.yaml"
        import yaml
        with open(config_snapshot, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.logger.debug(f"Configuration snapshot saved to: {config_snapshot}")

        # Get output format settings
        output_format = self.config.get('output', {}).get('format', 'coco').lower()
        annotation_type = self.config.get('output', {}).get('annotation_type', 'segmentation').lower()
        grayscale = self.config.get('output', {}).get('grayscale', False)

        self.logger.info(f"Output Format: {output_format.upper()}")
        self.logger.info(f"Annotation Type: {annotation_type.upper()}")
        self.logger.info(f"Grayscale Mode: {'ENABLED' if grayscale else 'DISABLED'}")

        # Initialize annotation files based on format
        coco_file_path = None
        labels_dir = None
        categories = create_categories()

        if output_format == 'yolo':
            # Get split configuration
            use_split = self.config.get('output', {}).get('use_split', True)
            train_ratio = self.config.get('output', {}).get('train_ratio', 0.7)
            val_ratio = self.config.get('output', {}).get('val_ratio', 0.2)
            test_ratio = self.config.get('output', {}).get('test_ratio', 0.1)
            split_seed = self.config.get('output', {}).get('split_seed', 42)

            # Initialize YOLO dataset structure
            images_dir, labels_dir, classes_file, dataset_yaml = initialize_yolo_dataset(
                str(base_output_dir),
                categories,
                use_split=use_split
            )
            self.logger.info(f"YOLO dataset initialized")
            self.logger.info(f"  - Images: {images_dir}")
            self.logger.info(f"  - Labels: {labels_dir}")
            self.logger.info(f"  - Classes: {classes_file}")
            self.logger.info(f"  - Dataset config: {dataset_yaml}")

            # Initialize split manager if using splits
            if use_split:
                # Collect all TIF files for deterministic split assignment
                from utils.geometry import find_tif_files
                all_tif_files = []
                for district in self.config['districts']:
                    mapdata_base_dir = self.config.get('mapdata_base_dir', 'datasets/MAPDATA')
                    tif_files = find_tif_files(district, mapdata_base_dir)
                    max_files = self.config['processing'].get('max_files_per_district')
                    if max_files and len(tif_files) > max_files:
                        tif_files = tif_files[:max_files]
                    all_tif_files.extend(tif_files)

                self.logger.info(f"  - Collected {len(all_tif_files)} TIF files for split assignment")

                initialize_split_manager(
                    str(base_output_dir),
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    seed=split_seed,
                    file_list=all_tif_files  # Pass file list for deterministic split
                )
                self.logger.info(f"  - Split: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

            # Still need COCO file path for intermediate processing
            coco_file_path = base_output_dir / "annotations" / "annotations_temp.json"
        else:
            # Initialize COCO JSON file with optional split support
            use_split = self.config.get('output', {}).get('use_split', False)

            # Enable FAST COCO mode (temp-file system)
            from utils.annotations import enable_temp_file_mode
            enable_temp_file_mode(str(base_output_dir), use_split=use_split)
            self.logger.info(f"[OK] Fast COCO mode enabled (temp-file system)")

            coco_file_path = Path(initialize_coco_file(str(base_output_dir), use_split=use_split))
            self.logger.info(f"COCO dataset initialized: {coco_file_path}")

            # Initialize split manager for COCO if using splits
            if use_split:
                train_ratio = self.config.get('output', {}).get('train_ratio', 0.7)
                val_ratio = self.config.get('output', {}).get('val_ratio', 0.2)
                test_ratio = self.config.get('output', {}).get('test_ratio', 0.1)
                split_seed = self.config.get('output', {}).get('split_seed', 42)

                # Collect all TIF files for deterministic split assignment
                from utils.geometry import find_tif_files
                all_tif_files = []
                for district in self.config['districts']:
                    mapdata_base_dir = self.config.get('mapdata_base_dir', 'datasets/MAPDATA')
                    tif_files = find_tif_files(district, mapdata_base_dir)
                    max_files = self.config['processing'].get('max_files_per_district')
                    if max_files and len(tif_files) > max_files:
                        tif_files = tif_files[:max_files]
                    all_tif_files.extend(tif_files)

                self.logger.info(f"  - Collected {len(all_tif_files)} TIF files for split assignment")

                initialize_split_manager(
                    str(base_output_dir),
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    seed=split_seed,
                    file_list=all_tif_files  # Pass file list for deterministic split
                )
                self.logger.info(f"  - Split: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

        return coco_file_path, labels_dir

    def configure_batch_processing(self):
        """Configure batch processing parameters."""
        batch_size = self.config.get('performance', {}).get('batch_size', 10)
        set_batch_size(batch_size)
        set_yolo_batch_size(batch_size)
        self.logger.info(f"Batch size: {batch_size}")

    def process_datasets(self, coco_file_path: Path):
        """
        Process datasets according to configuration.

        Args:
            coco_file_path: Path to COCO annotations file
        """
        base_output_dir = self.config['output_base_dir']

        # Process separate districts if enabled
        if self.config.get('processing_modes', {}).get('separate_districts', True):
            self.logger.info("=" * 70)
            self.logger.info("PROCESSING MODE: Separate Districts")
            self.logger.info("=" * 70)

            sep_images, sep_annotations = process_separate_districts(
                self.config,
                base_output_dir,
                str(coco_file_path)
            )

            self.total_images += sep_images
            self.total_annotations += sep_annotations

            self.logger.info(f"Separate districts complete: {sep_images} images, {sep_annotations} annotations")

        # Process combined maps if enabled
        if self.config.get('processing_modes', {}).get('combined_maps', True):
            self.logger.info("=" * 70)
            self.logger.info("PROCESSING MODE: Combined Maps")
            self.logger.info("=" * 70)

            comb_images, comb_annotations = process_combined_maps(
                self.config,
                base_output_dir,
                str(coco_file_path)
            )

            self.total_images += comb_images
            self.total_annotations += comb_annotations

            self.logger.info(f"Combined maps complete: {comb_images} images, {comb_annotations} annotations")

        # Merge temp files if using COCO format with temp-file mode
        output_format = self.config.get('output', {}).get('format', 'coco').lower()
        if output_format == 'coco':
            from utils.annotations import is_temp_file_mode_enabled, merge_all_split_temp_files, merge_temp_coco_files, disable_temp_file_mode
            from utils.config import create_categories

            if is_temp_file_mode_enabled():
                self.logger.info("=" * 70)
                self.logger.info("MERGING TEMP ANNOTATION FILES")
                self.logger.info("=" * 70)

                categories = create_categories()
                annotation_type = self.config.get('output', {}).get('annotation_type', 'segmentation').lower()
                use_split = self.config.get('output', {}).get('use_split', False)

                if use_split:
                    # Merge all split temp files
                    results = merge_all_split_temp_files(base_output_dir, categories, annotation_type)
                    for split, (img_count, ann_count) in results.items():
                        self.logger.info(f"  {split}: {img_count} images, {ann_count} annotations")
                else:
                    # Merge single temp directory
                    temp_dir = str(Path(base_output_dir) / 'temp_annotations')
                    final_coco_path = str(coco_file_path)
                    img_count, ann_count = merge_temp_coco_files(temp_dir, final_coco_path, categories, annotation_type)
                    self.logger.info(f"  Total: {img_count} images, {ann_count} annotations")

                disable_temp_file_mode()
                self.logger.info("✓ Merge complete!")

    def verify_output(self, coco_file_path: Path):
        """
        Verify generated output for consistency.

        Args:
            coco_file_path: Path to COCO annotations file
        """
        if not coco_file_path.exists():
            self.logger.warning(f"Annotations file not found: {coco_file_path}")
            return

        self.logger.info("=" * 70)
        self.logger.info("VERIFICATION")
        self.logger.info("=" * 70)

        # File size
        file_size_mb = coco_file_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Annotations file size: {file_size_mb:.1f} MB")

        # Verify COCO consistency
        try:
            import json
            with open(coco_file_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            num_images = len(coco_data['images'])
            num_annotations = len(coco_data['annotations'])
            num_categories = len(coco_data['categories'])

            self.logger.info(f"Images: {num_images}")
            self.logger.info(f"Annotations: {num_annotations}")
            self.logger.info(f"Categories: {num_categories}")

            # Check for orphaned annotations
            image_ids = {img['id'] for img in coco_data['images']}
            orphaned = [ann for ann in coco_data['annotations'] if ann['image_id'] not in image_ids]

            if orphaned:
                self.logger.warning(f"Found {len(orphaned)} orphaned annotations!")
            else:
                self.logger.info("[OK] All annotations have corresponding images")

            # Verify image files exist (sample)
            base_output_dir = Path(self.config['output_base_dir'])
            use_split = self.config.get('output', {}).get('use_split', False)

            missing_files = []
            for img in coco_data['images'][:10]:  # Check first 10
                if use_split:
                    # Check all split directories for the image (Coconuts-1 style: flat structure)
                    found = False
                    for split_name in ['train', 'valid', 'test']:
                        img_path = base_output_dir / split_name / img['file_name']
                        if img_path.exists():
                            found = True
                            break
                    if not found:
                        missing_files.append(img['file_name'])
                else:
                    # Check top-level images directory
                    images_dir = base_output_dir / 'images'
                    img_path = images_dir / img['file_name']
                    if not img_path.exists():
                        missing_files.append(img['file_name'])

            if missing_files:
                self.logger.warning(f"Missing {len(missing_files)} image files (sample check)")
                for missing in missing_files[:3]:
                    self.logger.warning(f"  - {missing}")
            else:
                self.logger.info("[OK] Image files verified (sample check)")

        except Exception as e:
            self.logger.error(f"Error during verification: {e}", exc_info=True)

        # Dimension verification
        use_split = self.config.get('output', {}).get('use_split', False)

        if use_split:
            # Verify all splits (Coconuts-1 style: _annotations.coco.json in split folders)
            total_issues = 0
            base_output_dir = Path(self.config['output_base_dir'])

            for split_name in ['train', 'valid', 'test']:
                split_coco_file = base_output_dir / split_name / '_annotations.coco.json'
                if split_coco_file.exists():
                    self.logger.info(f"Verifying {split_name} split...")
                    issues = verify_all_image_dimensions(
                        self.config['output_base_dir'],
                        str(split_coco_file)
                    )
                    total_issues += issues

            if total_issues == 0:
                self.logger.info("[OK] All image dimensions match metadata across all splits")
            else:
                self.logger.warning(f"Found {total_issues} total dimension mismatches across all splits")
        else:
            # Verify single annotations file
            dimension_issues = verify_all_image_dimensions(
                self.config['output_base_dir'],
                str(coco_file_path)
            )

            if dimension_issues == 0:
                self.logger.info("[OK] All image dimensions match metadata")
            else:
                self.logger.warning(f"Found {dimension_issues} dimension mismatches")

    def print_summary(self):
        """Print final processing summary."""
        self.logger.info("=" * 70)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("=" * 70)

        output_format = self.config.get('output', {}).get('format', 'coco')
        annotation_type = self.config.get('output', {}).get('annotation_type', 'segmentation')
        grayscale = self.config.get('output', {}).get('grayscale', False)

        self.logger.info(f"Districts processed: {len(self.config['districts'])}")
        self.logger.info(f"Total images: {self.total_images}")
        self.logger.info(f"Total annotations: {self.total_annotations}")
        self.logger.info(f"Output format: {output_format.upper()}")
        self.logger.info(f"Annotation type: {annotation_type.upper()}")
        self.logger.info(f"Grayscale: {'YES' if grayscale else 'NO'}")

        # Processing modes
        self.logger.info(f"Separate districts: {self.config.get('processing_modes', {}).get('separate_districts', False)}")
        self.logger.info(f"Combined maps: {self.config.get('processing_modes', {}).get('combined_maps', False)}")

        # Augmentations
        hue_enabled = self.config.get('hue_augmentation', {}).get('enabled', False)
        rotation_enabled = self.config.get('rotation', {}).get('enabled', False)

        self.logger.info(f"Hue augmentation: {hue_enabled}")
        if hue_enabled:
            self.logger.info(f"  - Variants per image: {self.config['hue_augmentation']['count']}")

        self.logger.info(f"Rotation: {rotation_enabled}")
        if rotation_enabled:
            self.logger.info(f"  - Rotations per image: {self.config['rotation']['count']}")

        # Output location
        base_output_dir = Path(self.config['output_base_dir'])
        self.logger.info(f"Output directory: {base_output_dir.absolute()}")

        if output_format == 'yolo':
            use_split = self.config.get('output', {}).get('use_split', True)
            if use_split:
                self.logger.info(f"  - Dataset config: {base_output_dir / 'dataset.yaml'}")
                # Print split statistics
                print_split_statistics(str(base_output_dir))
            else:
                self.logger.info(f"  - Labels: {base_output_dir / 'labels'}")
                self.logger.info(f"  - Dataset config: {base_output_dir / 'dataset.yaml'}")
        else:
            use_split = self.config.get('output', {}).get('use_split', False)
            if use_split:
                # Coconuts-1 style structure
                self.logger.info(f"  - Annotation files:")
                for split_name in ['train', 'valid', 'test']:
                    ann_file = base_output_dir / split_name / '_annotations.coco.json'
                    if ann_file.exists():
                        self.logger.info(f"    - {split_name}: {ann_file}")
            else:
                self.logger.info(f"  - Annotations: {base_output_dir / 'annotations' / 'annotations.json'}")

    def run(self) -> int:
        """
        Run the complete dataset generation pipeline.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            self.logger.info("Starting GIS Dataset Generator")
            self.logger.info("=" * 70)

            # Reset processing state to ensure clean counters for new run
            reset_processing_state()

            # Setup
            self.configure_batch_processing()
            coco_file_path, labels_dir = self.setup_output_directories()

            # Process
            self.process_datasets(coco_file_path)

            # Verify
            self.verify_output(coco_file_path)

            # Summary
            self.print_summary()

            self.logger.info("=" * 70)
            self.logger.info("[SUCCESS] Dataset generation completed successfully!")
            return 0

        except KeyboardInterrupt:
            self.logger.warning("\n" + "=" * 70)
            self.logger.warning("Process interrupted by user (Ctrl+C)")
            self.logger.warning("=" * 70)

            # Handle temp files if in COCO mode
            output_format = self.config.get('output', {}).get('format', 'coco').lower()
            if output_format == 'coco':
                from utils.annotations import is_temp_file_mode_enabled
                if is_temp_file_mode_enabled():
                    base_output_dir = Path(self.config['output_base_dir'])
                    use_split = self.config.get('output', {}).get('use_split', False)

                    self.logger.warning("⚠️  Temporary annotation files have been preserved")
                    self.logger.warning("   You can:")
                    self.logger.warning("   1. Re-run the script to continue from where it stopped")
                    self.logger.warning("   2. Manually merge temp files using:")

                    if use_split:
                        self.logger.warning(f"      python -c \"from utils.annotations import merge_all_split_temp_files; from utils.config import create_categories; merge_all_split_temp_files('{base_output_dir}', create_categories(), 'segmentation')\"")
                    else:
                        temp_dir = base_output_dir / 'temp_annotations'
                        final_path = base_output_dir / 'annotations' / 'annotations.json'
                        self.logger.warning(f"      python -c \"from utils.annotations import merge_temp_coco_files; from utils.config import create_categories; merge_temp_coco_files('{temp_dir}', '{final_path}', create_categories(), 'segmentation')\"")

                    self.logger.warning(f"   3. Delete temp files manually if you want to start fresh:")
                    if use_split:
                        self.logger.warning(f"      rmdir /s {base_output_dir}\\train\\temp_annotations")
                        self.logger.warning(f"      rmdir /s {base_output_dir}\\valid\\temp_annotations")
                        self.logger.warning(f"      rmdir /s {base_output_dir}\\test\\temp_annotations")
                    else:
                        self.logger.warning(f"      rmdir /s {base_output_dir}\\temp_annotations")

            return 130
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
            return 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="GIS COCO/YOLO Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python main.py

  # Use custom configuration
  python main.py --config my_config.yaml

  # Enable verbose logging
  python main.py --verbose

  # Combine options
  python main.py --config configs/custom.yaml --verbose
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=None,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose (DEBUG level) logging"
    )

    parser.add_argument(
        '--version',
        action='version',
        version='GIS Dataset Generator v2.1.0'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    generator = DatasetGenerator(
        config_path=args.config,
        verbose=args.verbose
    )

    return generator.run()


if __name__ == "__main__":
    sys.exit(main())
