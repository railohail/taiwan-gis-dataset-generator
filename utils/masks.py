"""
Mask region management for GIS annotation filtering.

This module provides:
- Persistent YAML-based mask region storage
- Mask region CRUD operations per TIF file
- Singleton database access pattern

Masks are used to exclude zoom-ins, legends, scale bars, and other
decorative elements from annotation generation.

Database structure:
    mask_database/masks.yaml containing:
        filename.tif:
            - {x: 100, y: 50, width: 200, height: 150}
            - {x: 2800, y: 2400, width: 300, height: 200}
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional


class MaskDatabase:
    """
    Manages mask region definitions stored in a YAML database.

    Database structure:
        mask_database/
            masks.yaml - Main database file with all mask definitions

    YAML format:
        filename.tif:
            - {x: 100, y: 50, width: 200, height: 150}
            - {x: 2800, y: 2400, width: 300, height: 200}
    """

    def __init__(self, database_dir: str = "mask_database"):
        """
        Initialize mask database.

        Args:
            database_dir: Directory to store mask database (default: mask_database/)
        """
        self.database_dir = database_dir
        self.database_file = os.path.join(database_dir, "masks.yaml")
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """Create database directory and file if they don't exist."""
        os.makedirs(self.database_dir, exist_ok=True)

        if not os.path.exists(self.database_file):
            # Create empty database
            with open(self.database_file, 'w', encoding='utf-8') as f:
                yaml.dump({}, f, default_flow_style=False, allow_unicode=True)

    def load_all_masks(self) -> Dict[str, List[Dict]]:
        """
        Load all mask definitions from database.

        Returns:
            Dictionary mapping TIF filenames to list of mask region dicts
        """
        try:
            with open(self.database_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except Exception as e:
            print(f"Warning: Failed to load mask database: {e}")
            return {}

    def save_all_masks(self, masks: Dict[str, List[Dict]]):
        """
        Save all mask definitions to database.

        Args:
            masks: Dictionary mapping TIF filenames to list of mask region dicts
        """
        try:
            with open(self.database_file, 'w', encoding='utf-8') as f:
                yaml.dump(masks, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"Error: Failed to save mask database: {e}")

    def get_masks(self, tif_filename: str) -> Optional[List[Dict]]:
        """
        Get mask regions for a specific TIF file.

        Args:
            tif_filename: Base filename of TIF (e.g., "map_001.tif")

        Returns:
            List of mask region dicts, or None if not found
            Each dict has keys: x, y, width, height (all in pixels)
        """
        # Normalize filename to just basename
        tif_filename = os.path.basename(tif_filename)

        all_masks = self.load_all_masks()
        return all_masks.get(tif_filename)

    def set_masks(self, tif_filename: str, mask_regions: List[Dict]):
        """
        Set mask regions for a specific TIF file.

        Args:
            tif_filename: Base filename of TIF (e.g., "map_001.tif")
            mask_regions: List of mask region dicts
                Each dict should have keys: x, y, width, height (all in pixels)
        """
        # Normalize filename to just basename
        tif_filename = os.path.basename(tif_filename)

        all_masks = self.load_all_masks()
        all_masks[tif_filename] = mask_regions
        self.save_all_masks(all_masks)

        print(f"  ✓ Saved {len(mask_regions)} mask region(s) for {tif_filename}")

    def delete_masks(self, tif_filename: str):
        """
        Delete mask regions for a specific TIF file.

        Args:
            tif_filename: Base filename of TIF (e.g., "map_001.tif")
        """
        # Normalize filename to just basename
        tif_filename = os.path.basename(tif_filename)

        all_masks = self.load_all_masks()
        if tif_filename in all_masks:
            del all_masks[tif_filename]
            self.save_all_masks(all_masks)
            print(f"  ✓ Deleted masks for {tif_filename}")
        else:
            print(f"  No masks found for {tif_filename}")

    def has_masks(self, tif_filename: str) -> bool:
        """
        Check if mask regions exist for a specific TIF file.

        Args:
            tif_filename: Base filename of TIF (e.g., "map_001.tif")

        Returns:
            True if masks exist, False otherwise
        """
        # Normalize filename to just basename
        tif_filename = os.path.basename(tif_filename)

        all_masks = self.load_all_masks()
        return tif_filename in all_masks

    def list_all_files(self) -> List[str]:
        """
        List all TIF filenames that have mask definitions.

        Returns:
            List of TIF filenames
        """
        all_masks = self.load_all_masks()
        return list(all_masks.keys())

    def get_stats(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with stats (total_files, total_masks, etc.)
        """
        all_masks = self.load_all_masks()
        total_files = len(all_masks)
        total_masks = sum(len(masks) for masks in all_masks.values())

        return {
            'total_files': total_files,
            'total_masks': total_masks,
            'database_file': self.database_file
        }


# Global instance (can be used across modules)
_default_mask_db = None


def get_mask_database(database_dir: str = "mask_database") -> MaskDatabase:
    """
    Get or create the default mask database instance.

    Args:
        database_dir: Directory for mask database

    Returns:
        MaskDatabase instance
    """
    global _default_mask_db
    if _default_mask_db is None:
        _default_mask_db = MaskDatabase(database_dir)
    return _default_mask_db


def filter_annotations_with_masks(
    annotations: List[Dict],
    mask_regions: List[Dict],
    scaled_transform,
    raster_transform,
    image_width: int,
    image_height: int
) -> tuple:
    """
    Filter annotations by subtracting mask regions from polygons.

    Mask regions (zoom-ins, legends, etc.) are subtracted from annotation
    polygons, creating holes where the masks overlap.

    Args:
        annotations: List of COCO-format annotation dicts
        mask_regions: List of mask region dicts with x, y, width, height in original raster coords
        scaled_transform: Affine transform for the scaled/processed image
        raster_transform: Affine transform for the original raster
        image_width: Width of the output image
        image_height: Height of the output image

    Returns:
        Tuple of (filtered_annotations, stats_dict)
        stats_dict has keys: 'kept', 'clipped', 'filtered'
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    stats = {'kept': 0, 'clipped': 0, 'filtered': 0}

    if not mask_regions or not annotations:
        stats['kept'] = len(annotations)
        return annotations, stats

    # Convert mask regions to scaled image pixel coordinates
    mask_polygons = []
    for region in mask_regions:
        # Mask regions are in original raster pixel coordinates
        x1_px, y1_px = region['x'], region['y']
        x2_px, y2_px = x1_px + region['width'], y1_px + region['height']

        # Convert original pixel coords to geographic coords
        x1_geo, y1_geo = raster_transform * (x1_px, y1_px)
        x2_geo, y2_geo = raster_transform * (x2_px, y2_px)

        # Convert geographic coords to scaled image pixel coords
        x1_scaled, y1_scaled = ~scaled_transform * (x1_geo, y1_geo)
        x2_scaled, y2_scaled = ~scaled_transform * (x2_geo, y2_geo)

        # Create mask box in scaled pixel coordinates
        mask_box = box(
            min(x1_scaled, x2_scaled),
            min(y1_scaled, y2_scaled),
            max(x1_scaled, x2_scaled),
            max(y1_scaled, y2_scaled)
        )
        mask_polygons.append(mask_box)

    # Combine all mask regions
    combined_mask = unary_union(mask_polygons)

    filtered_annotations = []
    for ann in annotations:
        if 'segmentation' not in ann or not ann['segmentation']:
            # No segmentation, keep as-is
            filtered_annotations.append(ann)
            stats['kept'] += 1
            continue

        # Convert segmentation to polygon(s)
        try:
            seg = ann['segmentation']
            if isinstance(seg, list) and len(seg) > 0:
                # COCO format: list of polygon coordinate lists
                if isinstance(seg[0], list):
                    # Multiple polygons - process largest
                    coords = seg[0]
                else:
                    coords = seg

                # Convert flat list to coordinate pairs
                points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

                if len(points) < 3:
                    stats['filtered'] += 1
                    continue

                poly = Polygon(points)

                if not poly.is_valid:
                    poly = poly.buffer(0)

                if poly.is_empty:
                    stats['filtered'] += 1
                    continue

                # Subtract mask regions from polygon
                if poly.intersects(combined_mask):
                    clipped_poly = poly.difference(combined_mask)

                    if clipped_poly.is_empty:
                        stats['filtered'] += 1
                        continue

                    # Update annotation with clipped polygon
                    new_ann = ann.copy()

                    # Handle MultiPolygon result - take largest
                    if clipped_poly.geom_type == 'MultiPolygon':
                        clipped_poly = max(clipped_poly.geoms, key=lambda g: g.area)

                    if clipped_poly.geom_type == 'Polygon' and not clipped_poly.is_empty:
                        # Convert back to COCO format
                        coords = list(clipped_poly.exterior.coords)
                        flat_coords = []
                        for x, y in coords[:-1]:  # Exclude closing point
                            flat_coords.extend([float(x), float(y)])

                        new_ann['segmentation'] = [flat_coords]

                        # Recalculate bbox and area
                        xs = flat_coords[0::2]
                        ys = flat_coords[1::2]
                        new_ann['bbox'] = [
                            min(xs),
                            min(ys),
                            max(xs) - min(xs),
                            max(ys) - min(ys)
                        ]
                        new_ann['area'] = clipped_poly.area

                        filtered_annotations.append(new_ann)
                        stats['clipped'] += 1
                    else:
                        stats['filtered'] += 1
                else:
                    # No intersection with mask, keep original
                    filtered_annotations.append(ann)
                    stats['kept'] += 1
            else:
                filtered_annotations.append(ann)
                stats['kept'] += 1

        except (ValueError, TypeError, IndexError):
            # If we can't process, keep original
            filtered_annotations.append(ann)
            stats['kept'] += 1

    return filtered_annotations, stats
