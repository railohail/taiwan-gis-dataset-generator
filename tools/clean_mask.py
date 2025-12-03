"""
Clean Mask Region Selector
---------------------------
Interactive tool to define mask regions (zoom-ins, legends, decorative elements)
that should be excluded from map annotations.

Usage:
    python -m tools.clean_mask

Features:
    - Draw rectangles on map images to define exclusion zones
    - Automatic saving to persistent mask database
    - Visual preview of how masks affect shapefile polygons
    - Works with both separate and combined processing modes

Controls:
    Click & Drag    - Draw mask rectangle
    'r' key         - Reset/clear all rectangles
    'n' key         - Next/save and visualize
    's' key         - Skip current image
    'q' key         - Quit

Output:
    mask_database/masks.yaml                    - Persistent mask database
    interact/mask_regions.yaml                  - Latest mask definitions (copy)
    interact/visualizations/*_mask_test.png     - Before/after visualization
"""

import cv2
import numpy as np
import yaml
import os
import sys
import glob
from pathlib import Path
import rasterio
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class MaskRegionSelector:
    """Interactive tool for selecting mask regions on map images."""

    def __init__(self, shapefile_path='datasets/shapefile/COUNTY_MOI_1130718.shp'):
        """
        Initialize the mask selector.

        Args:
            shapefile_path: Path to county boundary shapefile
        """
        self.rectangles = []
        self.drawing = False
        self.start_point = None
        self.current_image = None
        self.display_image = None
        self.shapefile_path = shapefile_path
        self.image_name = ""
        self.scale = 1.0
        self.original_size = None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing rectangles."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Draw preview rectangle
                self.display_image = self.current_image.copy()
                cv2.rectangle(self.display_image, self.start_point, (x, y), (0, 0, 255), 2)

                # Redraw existing rectangles
                for rect in self.rectangles:
                    cv2.rectangle(self.display_image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
                    overlay = self.display_image.copy()
                    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.2, self.display_image, 0.8, 0, self.display_image)

                cv2.imshow('Mask Region Selector', self.display_image)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)

                x1 = min(self.start_point[0], end_point[0])
                y1 = min(self.start_point[1], end_point[1])
                x2 = max(self.start_point[0], end_point[0])
                y2 = max(self.start_point[1], end_point[1])

                if x2 - x1 > 5 and y2 - y1 > 5:
                    self.rectangles.append((x1, y1, x2, y2))

                self.display_image = self.current_image.copy()
                for rect in self.rectangles:
                    cv2.rectangle(self.display_image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
                    overlay = self.display_image.copy()
                    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.2, self.display_image, 0.8, 0, self.display_image)

                cv2.imshow('Mask Region Selector', self.display_image)

    def load_tif_image(self, tif_path, max_display_size=1200):
        """Load TIF file and convert to displayable image."""
        try:
            with rasterio.open(tif_path) as raster:
                raster_data = raster.read()

                if len(raster_data.shape) == 3:
                    image = np.transpose(raster_data, (1, 2, 0))
                else:
                    image = raster_data

                if len(image.shape) == 3 and image.shape[2] >= 3:
                    image = image[:, :, :3]
                elif len(image.shape) == 2 or image.shape[2] == 1:
                    if len(image.shape) == 3:
                        gray = image[:, :, 0]
                    else:
                        gray = image
                    image = np.stack([gray, gray, gray], axis=2)

                if image.max() > 255:
                    image = (image / image.max() * 255).astype(np.uint8)
                elif image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

                original_height, original_width = image.shape[:2]

                scale = 1.0
                if original_width > max_display_size or original_height > max_display_size:
                    scale = min(max_display_size / original_width, max_display_size / original_height)
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                return image, scale, (original_width, original_height)

        except Exception as e:
            print(f"Error loading {tif_path}: {e}")
            return None, None, None

    def crop_image_and_shapefile(self, raster, image_data, shapefile, crop_factor):
        """Crop the raster image and clip shapefile to new bounds (combined mode)."""
        height, width = image_data.shape[:2]
        crop_x = int(width * crop_factor)
        crop_y = int(height * crop_factor)

        if (2 * crop_x >= width) or (2 * crop_y >= height):
            print("Error: Crop factor is too large.")
            return None, None, None, None

        cropped_image = image_data[crop_y:height-crop_y, crop_x:width-crop_x]

        transform = raster.transform
        cropped_transform = rasterio.Affine(
            transform.a, transform.b,
            transform.c + transform.a * crop_x + transform.b * crop_y,
            transform.d, transform.e,
            transform.f + transform.d * crop_x + transform.e * crop_y
        )

        cropped_bounds = rasterio.transform.array_bounds(
            cropped_image.shape[0], cropped_image.shape[1], cropped_transform
        )

        bbox = box(*cropped_bounds)
        bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs=raster.crs)
        clipped_shapefile = gpd.clip(shapefile, bbox_gdf)

        return cropped_image, cropped_transform, cropped_bounds, clipped_shapefile

    def apply_masks_to_shapefile(self, shapefile_gdf, mask_regions, raster_transform):
        """Subtract mask rectangles from shapefile polygons."""
        if not mask_regions or len(mask_regions) == 0:
            return shapefile_gdf, [], {}

        mask_polygons = []
        for region in mask_regions:
            x1_px, y1_px = region['x'], region['y']
            x2_px, y2_px = x1_px + region['width'], y1_px + region['height']

            x1_geo, y1_geo = raster_transform * (x1_px, y1_px)
            x2_geo, y2_geo = raster_transform * (x2_px, y2_px)

            mask_poly = box(
                min(x1_geo, x2_geo),
                min(y1_geo, y2_geo),
                max(x1_geo, x2_geo),
                max(y1_geo, y2_geo)
            )
            mask_polygons.append(mask_poly)

        combined_mask = unary_union(mask_polygons)

        # Track original hole counts before masking
        original_holes = {}
        for idx, row in shapefile_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                original_holes[idx] = len(geom.interiors)
            elif geom.geom_type == 'MultiPolygon':
                original_holes[idx] = sum(len(p.interiors) for p in geom.geoms)
            else:
                original_holes[idx] = 0

        masked_shapefile = shapefile_gdf.copy()
        masked_shapefile['geometry'] = masked_shapefile['geometry'].apply(
            lambda geom: geom.difference(combined_mask) if geom.intersects(combined_mask) else geom
        )

        masked_shapefile = masked_shapefile[~masked_shapefile.geometry.is_empty]

        return masked_shapefile, mask_polygons, original_holes

    def create_visualization(self, cropped_image, original_shapefile, masked_shapefile,
                           mask_regions, transform, tif_filename, original_transform=None, original_holes=None):
        """
        Create 3-panel before/after visualization.

        IMPORTANT: This is ONLY for visual inspection. The text labels and visualization
        do NOT affect the actual mask processing in main.py. The masks are applied to
        the shapefile geometry before annotations are generated.
        """
        print("\n  Creating visualization...")

        if cropped_image.max() > 255:
            image = (cropped_image / cropped_image.max() * 255).astype(np.uint8)
        elif cropped_image.max() <= 1.0:
            image = (cropped_image * 255).astype(np.uint8)
        else:
            image = cropped_image.astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        height, width = image.shape[:2]
        extent = [0, width, height, 0]

        import matplotlib.cm as cm
        num_counties = len(original_shapefile)
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_counties))

        # Convert mask regions to cropped image coordinates
        # Mask regions are in ORIGINAL image coords, need to convert to CROPPED coords
        adjusted_mask_regions = []
        if original_transform and mask_regions:
            for region in mask_regions:
                # Get geographic coordinates of mask corners
                x1_px_orig, y1_px_orig = region['x'], region['y']
                x2_px_orig, y2_px_orig = x1_px_orig + region['width'], y1_px_orig + region['height']

                # Convert original pixel coords to geographic coords
                x1_geo, y1_geo = original_transform * (x1_px_orig, y1_px_orig)
                x2_geo, y2_geo = original_transform * (x2_px_orig, y2_px_orig)

                # Convert geographic coords to cropped image pixel coords
                x1_px_crop, y1_px_crop = ~transform * (x1_geo, y1_geo)
                x2_px_crop, y2_px_crop = ~transform * (x2_geo, y2_geo)

                adjusted_mask_regions.append({
                    'x': int(min(x1_px_crop, x2_px_crop)),
                    'y': int(min(y1_px_crop, y2_px_crop)),
                    'width': int(abs(x2_px_crop - x1_px_crop)),
                    'height': int(abs(y2_px_crop - y1_px_crop))
                })
        else:
            adjusted_mask_regions = mask_regions

        # Panel 1: Original shapefile
        ax1 = axes[0]
        ax1.imshow(image, extent=extent)
        ax1.set_title('BEFORE: Original Shapefile\n(Combined Mode - All Counties)',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')

        for idx, row in original_shapefile.iterrows():
            geom = row.geometry
            if geom.geom_type == 'MultiPolygon':
                geom = max(geom.geoms, key=lambda x: x.area)

            if geom.geom_type == 'Polygon':
                exterior = geom.exterior.coords[:]
                pixel_coords = [~transform * (x, y) for x, y in exterior]
                xs, ys = zip(*pixel_coords)
                color = colors[idx % len(colors)]
                ax1.fill(xs, ys, alpha=0.3, fc=color, ec=color, linewidth=2)
                ax1.plot(xs, ys, color=color, linewidth=2)

        # Panel 2: Mask regions overlay
        ax2 = axes[1]
        ax2.imshow(image, extent=extent)
        ax2.set_title('Mask Regions (Red Rectangles)\n(Areas to Exclude from Annotations)',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')

        for idx, row in original_shapefile.iterrows():
            geom = row.geometry
            if geom.geom_type == 'MultiPolygon':
                geom = max(geom.geoms, key=lambda x: x.area)

            if geom.geom_type == 'Polygon':
                exterior = geom.exterior.coords[:]
                pixel_coords = [~transform * (x, y) for x, y in exterior]
                xs, ys = zip(*pixel_coords)
                ax2.plot(xs, ys, 'b-', linewidth=1, alpha=0.3)

        for region in adjusted_mask_regions:
            x1, y1 = region['x'], region['y']
            x2, y2 = x1 + region['width'], y1 + region['height']
            rect = mpatches.Rectangle((x1, y1), region['width'], region['height'],
                                     linewidth=4, edgecolor='red', facecolor='red', alpha=0.5)
            ax2.add_patch(rect)
            ax2.text(x1 + region['width']/2, y1 + region['height']/2, 'MASKED',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)

        # Panel 3: After - masked shapefile
        ax3 = axes[2]
        ax3.imshow(image, extent=extent)
        ax3.set_title('AFTER: Shapefile with Masks Applied\n(White/Red Dashed Areas = Masked Regions)',
                     fontsize=14, fontweight='bold', color='green')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')

        for idx, row in masked_shapefile.iterrows():
            geom = row.geometry

            if geom.geom_type == 'MultiPolygon':
                polygons = list(geom.geoms)
            elif geom.geom_type == 'Polygon':
                polygons = [geom]
            else:
                continue

            color = colors[idx % len(colors)]

            # Get original hole count for this county
            original_hole_count = original_holes.get(idx, 0) if original_holes else 0

            # Count current holes
            current_hole_count = 0
            if geom.geom_type == 'Polygon':
                current_hole_count = len(geom.interiors)
            elif geom.geom_type == 'MultiPolygon':
                current_hole_count = sum(len(p.interiors) for p in geom.geoms)

            # Only label NEW holes created by masking
            holes_created_by_masking = current_hole_count - original_hole_count

            hole_index = 0
            for poly in polygons:
                exterior = poly.exterior.coords[:]
                pixel_coords = [~transform * (x, y) for x, y in exterior]
                xs, ys = zip(*pixel_coords)
                ax3.fill(xs, ys, alpha=0.3, fc=color, ec=color, linewidth=2)
                ax3.plot(xs, ys, color=color, linewidth=2)

                for interior in poly.interiors:
                    hole_coords = interior.coords[:]
                    hole_pixel_coords = [~transform * (x, y) for x, y in hole_coords]
                    hole_xs, hole_ys = zip(*hole_pixel_coords)

                    # Determine if this is a new hole or pre-existing
                    is_new_hole = hole_index >= original_hole_count

                    if is_new_hole:
                        # Show NEW holes (created by masks) with white fill and red dashed border
                        # NO TEXT LABEL - just visual indication
                        ax3.fill(hole_xs, hole_ys, alpha=1.0, fc='white', ec='red', linewidth=2)
                        ax3.plot(hole_xs, hole_ys, 'r--', linewidth=2)
                    else:
                        # Show pre-existing holes normally (just outline, no label)
                        ax3.plot(hole_xs, hole_ys, color=color, linewidth=1, linestyle='-', alpha=0.5)

                    hole_index += 1

        for region in adjusted_mask_regions:
            x1, y1 = region['x'], region['y']
            x2, y2 = x1 + region['width'], y1 + region['height']
            rect = mpatches.Rectangle((x1, y1), region['width'], region['height'],
                                     linewidth=2, edgecolor='red', facecolor='none',
                                     linestyle='--', alpha=0.7)
            ax3.add_patch(rect)

        plt.tight_layout()

        os.makedirs('interact/visualizations', exist_ok=True)
        output_path = f"interact/visualizations/{Path(tif_filename).stem}_combined_mask_test.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved visualization: {output_path}")

        # Print statistics
        print(f"\n  Statistics:")
        print(f"    Image size: {width}x{height} pixels")
        print(f"    Original counties: {len(original_shapefile)}")
        print(f"    Counties after masking: {len(masked_shapefile)}")
        print(f"    Mask regions applied: {len(mask_regions)}")

        if 'COUNTYENG' in original_shapefile.columns:
            counties = original_shapefile['COUNTYENG'].unique().tolist()
            print(f"    Counties in image: {', '.join(counties)}")

        original_area = original_shapefile.geometry.area.sum()
        masked_area = masked_shapefile.geometry.area.sum()
        area_reduction = original_area - masked_area
        area_reduction_pct = (area_reduction / original_area * 100) if original_area > 0 else 0
        print(f"    Annotation area removed: {area_reduction_pct:.2f}%")

    def process_tif_file(self, tif_path):
        """Process a single TIF file interactively."""
        self.rectangles = []
        self.image_name = os.path.basename(tif_path)

        print(f"\nLoading: {self.image_name}")
        image, scale, original_size = self.load_tif_image(tif_path)

        if image is None:
            print(f"  Skipping (failed to load)")
            return None

        self.scale = scale
        self.original_size = original_size
        self.current_image = image.copy()
        self.display_image = image.copy()

        cv2.namedWindow('Mask Region Selector', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Mask Region Selector', self.mouse_callback)

        instructions = [
            "Draw rectangles around areas to MASK (zoom-ins, legends, etc.)",
            "Controls: [r] reset | [n] next/save+visualize | [s] skip | [q] quit",
            f"Image: {self.image_name} | Scale: {scale:.2f}x"
        ]

        display = self.display_image.copy()
        y_offset = 30
        for line in instructions:
            cv2.putText(display, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30

        cv2.imshow('Mask Region Selector', display)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.rectangles = []
                self.display_image = self.current_image.copy()
                cv2.imshow('Mask Region Selector', self.display_image)
                print("  Reset rectangles")

            elif key == ord('n'):
                if len(self.rectangles) > 0:
                    original_rects = []
                    for rect in self.rectangles:
                        x1 = int(rect[0] / scale)
                        y1 = int(rect[1] / scale)
                        x2 = int(rect[2] / scale)
                        y2 = int(rect[3] / scale)
                        original_rects.append({
                            'x': x1,
                            'y': y1,
                            'width': x2 - x1,
                            'height': y2 - y1
                        })

                    print(f"  Saved {len(original_rects)} mask region(s)")
                    return original_rects
                else:
                    print("  No masks defined, skipping")
                    return None

            elif key == ord('s'):
                print("  Skipped")
                return None

            elif key == ord('q'):
                print("\nQuitting...")
                return 'QUIT'

    def run(self, tif_file_path, crop_factor=0.05):
        """Run interactive mask selection and visualization (combined mode)."""
        print("=" * 70)
        print("Clean Mask Region Selector")
        print("(COMBINED MODE - Full map with all counties)")
        print("=" * 70)

        if not os.path.exists(tif_file_path):
            print(f"\nError: TIF file not found: {tif_file_path}")
            return

        if not os.path.exists(self.shapefile_path):
            print(f"\nError: Shapefile not found: {self.shapefile_path}")
            return

        os.makedirs('interact', exist_ok=True)
        os.makedirs('interact/visualizations', exist_ok=True)

        print(f"\nProcessing: {tif_file_path}")
        print(f"Shapefile: {self.shapefile_path}")
        print(f"Crop factor: {crop_factor}")

        # Interactive mask selection
        mask_regions = self.process_tif_file(tif_file_path)

        if mask_regions == 'QUIT':
            cv2.destroyAllWindows()
            return

        cv2.destroyAllWindows()

        if mask_regions is None:
            print("\nNo masks defined. Exiting.")
            return

        # Load and process combined mode
        print("\nLoading raster and shapefile (combined mode)...")
        with rasterio.open(tif_file_path) as raster:
            shapefile = gpd.read_file(self.shapefile_path)
            if shapefile.crs != raster.crs:
                shapefile = shapefile.to_crs(raster.crs)

            print(f"  Loaded {len(shapefile)} county polygons")

            # Capture original transform BEFORE cropping (needed for coordinate conversion)
            original_transform = raster.transform

            raster_data = raster.read()
            if len(raster_data.shape) == 3:
                image_data = np.transpose(raster_data, (1, 2, 0))
            else:
                image_data = raster_data

            if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
                image_data = image_data[:, :, :3]
            elif len(image_data.shape) == 2 or image_data.shape[2] == 1:
                if len(image_data.shape) == 3:
                    gray = image_data[:, :, 0]
                else:
                    gray = image_data
                image_data = np.stack([gray, gray, gray], axis=2)

            if image_data.max() > 255:
                image_data = (image_data / image_data.max() * 255).astype(np.uint8)
            elif image_data.max() <= 1.0:
                image_data = (image_data * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)

            print(f"  Cropping image with crop_factor={crop_factor}...")
            cropped_result = self.crop_image_and_shapefile(raster, image_data, shapefile, crop_factor)

            if any(x is None for x in cropped_result):
                print("  ERROR: Failed to crop image")
                return

            cropped_image, cropped_transform, cropped_bounds, clipped_shapefile = cropped_result
            print(f"  Cropped to: {cropped_image.shape[1]}x{cropped_image.shape[0]} pixels")
            print(f"  Counties in cropped area: {len(clipped_shapefile)}")

        # Apply masks to shapefile (use original_transform since mask_regions are in original coords)
        print("\nApplying masks to shapefile...")
        masked_shapefile, mask_polygons, original_holes = self.apply_masks_to_shapefile(
            clipped_shapefile, mask_regions, original_transform
        )

        # Create visualization (pass original_transform for coordinate conversion)
        self.create_visualization(
            cropped_image, clipped_shapefile, masked_shapefile,
            mask_regions, cropped_transform, os.path.basename(tif_file_path),
            original_transform=original_transform,
            original_holes=original_holes
        )

        # Save to mask database
        from utils.masks import get_mask_database

        mask_db = get_mask_database()
        tif_basename = os.path.basename(tif_file_path)

        if mask_db.has_masks(tif_basename):
            print(f"\n⚠  Masks already exist for {tif_basename}")
            print("   Auto-overwriting...")

        mask_db.set_masks(tif_basename, mask_regions)

        # Save copy to interact folder
        output_path = os.path.join('interact', 'mask_regions.yaml')
        mask_config = {
            'mask_definitions': {
                tif_basename: mask_regions
            },
            'info': {
                'format': 'regions are in original image coordinates (pixels)',
                'usage': 'Masks will subtract from shapefile polygons (create holes)',
                'mode': 'combined - applies to full map with all counties',
                'note': 'Saved in mask_database/ and auto-applied by main.py'
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(mask_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        stats = mask_db.get_stats()

        print(f"\n✓ Saved to database: {stats['database_file']}")
        print(f"✓ Copy saved to: {output_path}")
        print(f"\nMask Database Stats:")
        print(f"  Total files with masks: {stats['total_files']}")
        print(f"  Total mask regions: {stats['total_masks']}")
        print("\n" + "=" * 70)
        print("Done! Check 'interact/visualizations/' for before/after comparison.")
        print("\nTo remove these masks:")
        print(f"  from utils.masks import get_mask_database")
        print(f"  mask_db = get_mask_database()")
        print(f"  mask_db.delete_masks('{tif_basename}')")
        print("=" * 70)


def get_folder_from_path(tif_path):
    """Get the district folder name from TIF path."""
    # Get relative path from MAPDATA
    rel_path = os.path.relpath(tif_path, 'datasets/MAPDATA')
    # Get first component (folder name)
    folder = rel_path.split(os.sep)[0]
    return folder


def select_folder_filter(tif_files):
    """Let user select which folders to process."""
    # Get unique folders
    folders = {}
    for tif_file in tif_files:
        folder = get_folder_from_path(tif_file)
        if folder not in folders:
            folders[folder] = []
        folders[folder].append(tif_file)

    print("\n" + "=" * 70)
    print("Available folders (districts):")
    print("=" * 70)

    folder_list = sorted(folders.keys())
    for idx, folder in enumerate(folder_list, 1):
        count = len(folders[folder])
        print(f"  [{idx}] {folder} ({count} file{'s' if count != 1 else ''})")

    print("\n" + "=" * 70)
    print("Select folders to process:")
    print("  - Enter folder number(s) separated by comma (e.g., 1,3,5)")
    print("  - Press 'a' to include ALL folders")
    print("  - Press 'q' to quit")
    print("=" * 70)

    while True:
        choice = input("\nYour choice: ").strip().lower()

        if choice == 'q':
            return None
        elif choice == 'a':
            return list(tif_files)
        else:
            try:
                # Parse comma-separated numbers
                indices = [int(x.strip()) for x in choice.split(',')]
                selected_folders = []

                for idx in indices:
                    if 1 <= idx <= len(folder_list):
                        selected_folders.append(folder_list[idx - 1])
                    else:
                        print(f"Invalid number: {idx}. Please enter 1-{len(folder_list)}")
                        selected_folders = None
                        break

                if selected_folders:
                    # Get all files from selected folders
                    filtered_files = []
                    for folder in selected_folders:
                        filtered_files.extend(folders[folder])

                    print(f"\nSelected {len(selected_folders)} folder(s): {', '.join(selected_folders)}")
                    print(f"Total files: {len(filtered_files)}")
                    return filtered_files

            except ValueError:
                print("Invalid input. Please enter comma-separated numbers, 'a', or 'q'")


def select_file_interactive(tif_files):
    """Let user select which file to process."""
    print("\n" + "=" * 70)
    print("Available TIF files:")
    print("=" * 70)

    for idx, tif_file in enumerate(tif_files, 1):
        # Show relative path from datasets/MAPDATA
        rel_path = os.path.relpath(tif_file, 'datasets/MAPDATA')
        print(f"  [{idx}] {rel_path}")

    print("\n" + "=" * 70)
    print("Select file to process:")
    print("  - Enter number (1-{})".format(len(tif_files)))
    print("  - Press 'a' to process ALL files")
    print("  - Press 'b' to go back to folder selection")
    print("  - Press 'q' to quit")
    print("=" * 70)

    while True:
        choice = input("\nYour choice: ").strip().lower()

        if choice == 'q':
            return None
        elif choice == 'a':
            return 'ALL'
        elif choice == 'b':
            return 'BACK'
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(tif_files):
                    return tif_files[idx - 1]
                else:
                    print(f"Invalid number. Please enter 1-{len(tif_files)}")
            except ValueError:
                print("Invalid input. Please enter a number, 'a', 'b', or 'q'")


def main():
    """Main entry point for the clean mask tool."""
    # Find all TIF files
    all_tif_files = glob.glob('datasets/MAPDATA/**/*.tif', recursive=True)
    all_tif_files = [f for f in all_tif_files if not f.endswith('.aux.xml')]

    if not all_tif_files:
        print("No TIF files found in datasets/MAPDATA/")
        print("Please place TIF files there first.")
        return 1

    print(f"\nFound {len(all_tif_files)} TIF file(s)")

    # Step 1: Select folders
    filtered_files = select_folder_filter(all_tif_files)

    if filtered_files is None:
        print("Exiting...")
        return 0

    # Step 2: Select specific file(s) from filtered list
    while True:
        selection = select_file_interactive(filtered_files)

        if selection is None:
            print("Exiting...")
            break
        elif selection == 'BACK':
            # Go back to folder selection
            filtered_files = select_folder_filter(all_tif_files)
            if filtered_files is None:
                print("Exiting...")
                break
            continue
        elif selection == 'ALL':
            # Process all filtered files
            print("\n" + "=" * 70)
            print(f"Processing ALL {len(filtered_files)} files")
            print("=" * 70)

            selector = MaskRegionSelector(shapefile_path='datasets/shapefile/COUNTY_MOI_1130718.shp')

            for idx, tif_file in enumerate(filtered_files, 1):
                print(f"\n\n[File {idx}/{len(filtered_files)}]")
                print("=" * 70)

                # Check if already has masks
                from utils.masks import get_mask_database
                mask_db = get_mask_database()
                tif_basename = os.path.basename(tif_file)

                if mask_db.has_masks(tif_basename):
                    print(f"⚠  Masks already exist for: {tif_basename}")
                    print("   Options: [s]kip | [o]verwrite | [q]uit")
                    action = input("   Your choice: ").strip().lower()

                    if action == 's':
                        print("   Skipped.")
                        continue
                    elif action == 'q':
                        print("\nStopping batch processing.")
                        break
                    elif action == 'o':
                        print("   Overwriting...")
                    else:
                        print("   Skipped (invalid input).")
                        continue

                selector.run(tif_file, crop_factor=0.05)

            print("\n" + "=" * 70)
            print("Batch processing complete!")
            print("=" * 70)
            break
        else:
            # Single file processing
            print(f"\nProcessing: {selection}\n")
            selector = MaskRegionSelector(shapefile_path='datasets/shapefile/COUNTY_MOI_1130718.shp')
            selector.run(selection, crop_factor=0.05)
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
