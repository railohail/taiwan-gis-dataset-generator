"""
Visualization and verification utilities.

This module provides:
- Annotation verification against image bounds
- Debug overlay creation for annotation visualization
- Transparent mask visualization
- Batch dimension verification
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import cv2
import os
import json


def verify_annotations(image, annotations, image_name, verbose=False):
    """Verify that annotations are correctly aligned with the image.

    Args:
        image: Image data
        annotations: List of COCO annotations
        image_name: Name of the image
        verbose: If True, print detailed verification info (default False for performance)

    Returns:
        True if all annotations are valid
    """
    image_height, image_width = image.shape[:2]
    issues = []
    valid_annotations = 0

    if verbose:
        print(f"    Verifying {len(annotations)} annotations for {image_name} (image: {image_width}x{image_height})")

    for idx, ann in enumerate(annotations):
        segmentation = ann['segmentation'][0]
        bbox = ann['bbox']
        category_id = ann['category_id']

        # Check segmentation coordinates
        seg_issues = 0
        for i in range(0, len(segmentation), 2):
            x, y = segmentation[i], segmentation[i+1]
            if x < 0 or x >= image_width or y < 0 or y >= image_height:
                issues.append(f"Ann {idx} (cat {category_id}): Seg point ({x:.1f}, {y:.1f}) outside image bounds")
                seg_issues += 1

        # Check bbox coordinates
        x, y, w, h = bbox
        if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
            issues.append(f"Ann {idx} (cat {category_id}): Bbox ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}) outside image bounds")

        # Verify bbox matches segmentation
        seg_points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
        if seg_points:
            seg_x_coords = [p[0] for p in seg_points]
            seg_y_coords = [p[1] for p in seg_points]
            calc_bbox = [min(seg_x_coords), min(seg_y_coords),
                        max(seg_x_coords) - min(seg_x_coords),
                        max(seg_y_coords) - min(seg_y_coords)]

            bbox_diff = [abs(bbox[i] - calc_bbox[i]) for i in range(4)]
            if any(diff > 2.0 for diff in bbox_diff):  # Allow small floating point differences
                issues.append(f"Ann {idx} (cat {category_id}): Bbox mismatch - stored {bbox} vs calculated {calc_bbox}")

        if seg_issues == 0:
            valid_annotations += 1

    if issues and verbose:
        print(f"    WARNING: {len(issues)} annotation issues in {image_name}:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"      - {issue}")
        if len(issues) > 5:
            print(f"      - ... and {len(issues) - 5} more issues")

    if verbose:
        print(f"    Verification result: {valid_annotations}/{len(annotations)} annotations valid")
    return len(issues) == 0


def create_debug_annotation_overlay(image, annotations, categories, output_path):
    """Create a detailed debug visualization showing annotation accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Left side: Original image with segmentation overlays
    ax1.imshow(image)
    ax1.set_title("Segmentation Overlays")

    # Right side: Original image with bbox overlays
    ax2.imshow(image)
    ax2.set_title("Bounding Box Overlays")

    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

    for idx, ann in enumerate(annotations):
        category_id = ann['category_id']
        segmentation = ann['segmentation'][0]
        bbox = ann['bbox']

        color = colors[category_id % len(colors)]

        # Draw segmentation on left
        if len(segmentation) >= 6:
            points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
            polygon_patch = Polygon(points, facecolor=color, edgecolor='red', alpha=0.3, linewidth=2)
            ax1.add_patch(polygon_patch)

            # Add annotation index
            if points:
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)
                ax1.text(cx, cy, str(idx), ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Draw bbox on right
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2, alpha=0.8)
        ax2.add_patch(rect)
        ax2.text(x + w/2, y + h/2, str(idx), ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()

    try:
        debug_path = output_path.replace('.png', '_debug.png')
        plt.savefig(debug_path, dpi=150, bbox_inches='tight', facecolor='white')
    except Exception as e:
        from tqdm import tqdm
        tqdm.write(f"    Error saving debug visualization: {e}")

    plt.close()


def create_shapefile_overlay_visualization(image, shapefile, transform, output_path, title):
    """Create visualization showing shapefile polygons overlaid on the map (like clean_mask.py).

    Args:
        image: Image data (numpy array)
        shapefile: GeoDataFrame with county geometries
        transform: Rasterio affine transform for coordinate conversion
        output_path: Path to save visualization
        title: Title for the visualization
    """
    # Prepare image for display (EXACT copy from clean_mask.py)
    if image.max() > 255:
        display_image = (image / image.max() * 255).astype(np.uint8)
    elif image.max() <= 1.0:
        display_image = (image * 255).astype(np.uint8)
    else:
        display_image = image.astype(np.uint8)

    if display_image.ndim == 2:
        display_image = np.stack([display_image, display_image, display_image], axis=2)

    height, width = display_image.shape[:2]
    extent = [0, width, height, 0]

    # Create figure (single panel, not 3 panels like clean_mask.py)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(display_image, extent=extent)

    # Use tab20 colormap (EXACT copy from clean_mask.py line 247-249)
    import matplotlib.cm as cm
    num_counties = len(shapefile)
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_counties))

    # Draw each county polygon (EXACT copy from clean_mask.py lines 334-365)
    for idx, row in shapefile.iterrows():
        geom = row.geometry

        if geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)
        elif geom.geom_type == 'Polygon':
            polygons = [geom]
        else:
            continue

        color = colors[idx % len(colors)]

        for poly in polygons:
            exterior = poly.exterior.coords[:]
            pixel_coords = [~transform * (x, y) for x, y in exterior]
            xs, ys = zip(*pixel_coords)
            ax.fill(xs, ys, alpha=0.3, fc=color, ec=color, linewidth=2)
            ax.plot(xs, ys, color=color, linewidth=2)

            # Draw interior holes (if any)
            for interior in poly.interiors:
                hole_coords = interior.coords[:]
                hole_pixel_coords = [~transform * (x, y) for x, y in hole_coords]
                hole_xs, hole_ys = zip(*hole_pixel_coords)
                ax.plot(hole_xs, hole_ys, color=color, linewidth=1, linestyle='-', alpha=0.5)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Use English-only title to avoid font issues
    safe_title = title.encode('ascii', 'ignore').decode('ascii')
    plt.title(f"{safe_title}\n({len(shapefile)} counties)", fontsize=12, pad=20)
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    except Exception as e:
        from tqdm import tqdm
        tqdm.write(f"    Error saving shapefile visualization: {e}")

    plt.close()


def create_transparent_mask_visualization(image, annotations, categories, output_path, title, create_debug=False):
    """Create visualization with transparent colored masks (EXACT LEGACY VERSION).

    Args:
        image: Image data
        annotations: List of COCO annotations
        categories: List of COCO categories
        output_path: Path to save visualization
        title: Title for the visualization
        create_debug: If True, create debug overlay (slower, default False)
    """
    # Verify annotations first (non-verbose for performance)
    is_valid = verify_annotations(image, annotations, os.path.basename(output_path), verbose=False)

    # Create debug overlay for detailed verification (optional)
    if create_debug:
        create_debug_annotation_overlay(image, annotations, categories, output_path)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    n_categories = len(categories)
    # FIX: Set3 only has 12 colors but Taiwan has 22 counties!
    # Use tab20 (20 colors) + tab20b (20 colors) for full coverage
    if n_categories <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
    else:
        # Combine tab20 (0-19) + tab20b (20+) for 40 total colors
        colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, n_categories - 20))
        colors = np.vstack([colors_tab20, colors_tab20b])

    annotation_count = 0
    valid_annotations = 0

    for ann in annotations:
        category_id = ann['category_id']
        segmentation_list = ann['segmentation']

        if not segmentation_list:
            continue

        category_name = next((cat['name'] for cat in categories if cat['id'] == category_id), f"Category_{category_id}")
        image_height, image_width = image.shape[:2]

        # Process all polygons in the segmentation (handles multi-part annotations)
        all_points = []
        points_valid = True

        for seg_idx, segmentation in enumerate(segmentation_list):
            if len(segmentation) < 6:
                continue

            points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
            if not points:
                continue

            # Check if all points are within image bounds
            if not all(0 <= x < image_width and 0 <= y < image_height for x, y in points):
                points_valid = False

            # First polygon is exterior, subsequent polygons are holes or additional parts
            if seg_idx == 0:
                all_points = points
                polygon_patch = Polygon(points,
                                       facecolor=colors[category_id % len(colors)],
                                       edgecolor='red' if not points_valid else 'black',
                                       alpha=0.6,
                                       linewidth=2 if not points_valid else 1)
                ax.add_patch(polygon_patch)
            else:
                # Additional polygons (could be holes or disconnected parts)
                # Draw as white (hole) or same color (part) depending on winding
                # For simplicity, draw with reduced alpha to indicate additional parts
                polygon_patch = Polygon(points,
                                       facecolor='white',  # Holes shown as white
                                       edgecolor='gray',
                                       alpha=0.8,
                                       linewidth=1)
                ax.add_patch(polygon_patch)

        if all_points:
            cx = sum(p[0] for p in all_points) / len(all_points)
            cy = sum(p[1] for p in all_points) / len(all_points)
            # Use English name only to avoid font issues
            english_name = category_name.replace('County', 'Co').replace('City', 'Ci')

            # Show annotation info
            text_color = 'red' if not points_valid else 'black'
            ax.text(cx, cy, f"{english_name}\n(#{annotation_count})",
                   ha='center', va='center',
                   fontsize=6, fontweight='bold',
                   color=text_color,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

            annotation_count += 1
            if points_valid:
                valid_annotations += 1

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Use English-only title to avoid font issues
    safe_title = title.encode('ascii', 'ignore').decode('ascii')
    status = "VALID" if is_valid else "ISSUES"
    plt.title(f"{safe_title}\n({valid_annotations}/{annotation_count} annotations valid) - {status}",
             fontsize=12, pad=20, color='green' if is_valid else 'red')
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    except Exception as e:
        from tqdm import tqdm
        tqdm.write(f"    Error saving visualization: {e}")

    plt.close()

    return is_valid


def create_mask_triptych_visualization(image, original_shapefile, masked_shapefile,
                                       mask_regions, image_transform, output_path,
                                       raster_transform=None, title_prefix="Mask Debug"):
    """Replicate clean_mask.py's 3-panel before/mask/after visualization inside the pipeline."""
    if original_shapefile is None or masked_shapefile is None:
        return False

    if not mask_regions or len(mask_regions) == 0:
        return False

    if image_transform is None or raster_transform is None:
        from tqdm import tqdm
        tqdm.write("    Skipping mask triptych: missing transform information")
        return False

    # Prepare image for display
    display_image = image
    if display_image.max() > 255:
        display_image = (display_image / display_image.max() * 255).astype(np.uint8)
    elif display_image.max() <= 1.0:
        display_image = (display_image * 255).astype(np.uint8)
    else:
        display_image = display_image.astype(np.uint8)

    if display_image.ndim == 2:
        display_image = np.stack([display_image, display_image, display_image], axis=2)

    height, width = display_image.shape[:2]
    extent = [0, width, height, 0]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=16, fontweight='bold')

    cmap = cm.get_cmap('tab20')
    num_polygons = max(len(original_shapefile), 1)
    colors = cmap(np.linspace(0, 1, num_polygons))

    # Track original hole counts so we can highlight new holes created by masks
    original_holes = {}
    for idx, row in original_shapefile.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            original_holes[idx] = len(geom.interiors)
        elif geom.geom_type == 'MultiPolygon':
            original_holes[idx] = sum(len(poly.interiors) for poly in geom.geoms)
        else:
            original_holes[idx] = 0

    # Convert mask rectangles from original pixel coords to current image coords
    adjusted_mask_regions = []
    for region in mask_regions:
        x1_px_orig, y1_px_orig = region['x'], region['y']
        x2_px_orig = x1_px_orig + region['width']
        y2_px_orig = y1_px_orig + region['height']

        x1_geo, y1_geo = raster_transform * (x1_px_orig, y1_px_orig)
        x2_geo, y2_geo = raster_transform * (x2_px_orig, y2_px_orig)

        x1_px_img, y1_px_img = ~image_transform * (x1_geo, y1_geo)
        x2_px_img, y2_px_img = ~image_transform * (x2_geo, y2_geo)

        adjusted_mask_regions.append({
            'x': int(min(x1_px_img, x2_px_img)),
            'y': int(min(y1_px_img, y2_px_img)),
            'width': int(abs(x2_px_img - x1_px_img)),
            'height': int(abs(y2_px_img - y1_px_img))
        })

    # Panel 1: Original shapefile
    ax1 = axes[0]
    ax1.imshow(display_image, extent=extent)
    ax1.set_title("BEFORE: Original Shapefile", fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')

    for idx, row in original_shapefile.iterrows():
        geom = row.geometry
        geometries = []
        if geom.geom_type == 'MultiPolygon':
            geometries = list(geom.geoms)
        elif geom.geom_type == 'Polygon':
            geometries = [geom]
        else:
            continue

        color = colors[idx % len(colors)]
        for poly in geometries:
            exterior = poly.exterior.coords[:]
            pixel_coords = [~image_transform * (x, y) for x, y in exterior]
            xs, ys = zip(*pixel_coords)
            ax1.fill(xs, ys, alpha=0.3, fc=color, ec=color, linewidth=2)
            ax1.plot(xs, ys, color=color, linewidth=2)

    # Panel 2: Mask regions overlay
    ax2 = axes[1]
    ax2.imshow(display_image, extent=extent)
    ax2.set_title("Mask Regions (Red)", fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')

    for idx, row in original_shapefile.iterrows():
        geom = row.geometry
        geometries = []
        if geom.geom_type == 'MultiPolygon':
            geometries = list(geom.geoms)
        elif geom.geom_type == 'Polygon':
            geometries = [geom]
        else:
            continue

        for poly in geometries:
            exterior = poly.exterior.coords[:]
            pixel_coords = [~image_transform * (x, y) for x, y in exterior]
            xs, ys = zip(*pixel_coords)
            ax2.plot(xs, ys, 'b-', linewidth=1, alpha=0.3)

    for region in adjusted_mask_regions:
        rect = mpatches.Rectangle(
            (region['x'], region['y']),
            region['width'],
            region['height'],
            linewidth=4,
            edgecolor='red',
            facecolor='red',
            alpha=0.4
        )
        ax2.add_patch(rect)
        ax2.text(
            region['x'] + region['width'] / 2,
            region['y'] + region['height'] / 2,
            'MASKED',
            ha='center',
            va='center',
            color='white',
            fontweight='bold',
            fontsize=10
        )

    # Panel 3: Masked shapefile
    ax3 = axes[2]
    ax3.imshow(display_image, extent=extent)
    ax3.set_title("AFTER: Masks Applied", fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')

    for idx, row in masked_shapefile.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)
        elif geom.geom_type == 'Polygon':
            polygons = [geom]
        else:
            continue

        color = colors[idx % len(colors)]
        original_hole_count = original_holes.get(idx, 0)
        hole_index = 0

        for poly in polygons:
            exterior = poly.exterior.coords[:]
            pixel_coords = [~image_transform * (x, y) for x, y in exterior]
            xs, ys = zip(*pixel_coords)
            ax3.fill(xs, ys, alpha=0.3, fc=color, ec=color, linewidth=2)
            ax3.plot(xs, ys, color=color, linewidth=2)

            for interior in poly.interiors:
                hole_coords = interior.coords[:]
                hole_pixels = [~image_transform * (x, y) for x, y in hole_coords]
                hole_xs, hole_ys = zip(*hole_pixels)

                is_new_hole = hole_index >= original_hole_count
                if is_new_hole:
                    ax3.fill(hole_xs, hole_ys, alpha=1.0, fc='white', ec='red', linewidth=2)
                    ax3.plot(hole_xs, hole_ys, 'r--', linewidth=2)
                else:
                    ax3.plot(hole_xs, hole_ys, color=color, linewidth=1, linestyle='-', alpha=0.5)

                hole_index += 1

    for region in adjusted_mask_regions:
        rect = mpatches.Rectangle(
            (region['x'], region['y']),
            region['width'],
            region['height'],
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            alpha=0.7
        )
        ax3.add_patch(rect)

    for axis in [ax1, ax2, ax3]:
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect('equal')

    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        saved = True
    except Exception as e:
        from tqdm import tqdm
        tqdm.write(f"    Error saving mask triptych visualization: {e}")
        saved = False

    plt.close()
    return saved


def verify_all_image_dimensions(base_output_dir, coco_file_path):
    """Verify all saved images match their COCO metadata dimensions.

    Args:
        base_output_dir: Base output directory (may contain split structure)
        coco_file_path: Path to COCO JSON file
            - Coconuts-1 style: train/_annotations.coco.json
            - Old style: train/annotations/instances_train.json
            - No split: annotations/annotations.json
    """
    if not os.path.exists(coco_file_path):
        print("COCO file not found for dimension verification")
        return -1

    try:
        with open(coco_file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Determine if using split structure based on coco_file_path
        # Check for Coconuts-1 style (_annotations.coco.json) or old style (train/annotations/instances_train.json)
        coco_file_path_norm = os.path.normpath(coco_file_path)
        split_dir = None

        # Check for Coconuts-1 style: _annotations.coco.json directly in split folder
        if '_annotations.coco.json' in coco_file_path:
            parent_dir = os.path.basename(os.path.dirname(coco_file_path))
            if parent_dir in ['train', 'valid', 'test']:
                split_dir = parent_dir
        else:
            # Check for old style structure
            for split_name in ['train', 'val', 'valid', 'test']:
                if os.path.sep + split_name + os.path.sep in coco_file_path_norm or \
                   coco_file_path_norm.endswith(os.path.sep + split_name):
                    split_dir = split_name
                    break

        # Determine images directory
        if split_dir:
            # For Coconuts-1 style: images directly in split folder
            if '_annotations.coco.json' in coco_file_path:
                images_dir = os.path.join(base_output_dir, split_dir)
            else:
                # Old style: images in subdirectory
                images_dir = os.path.join(base_output_dir, split_dir, 'images')
        else:
            # Using single directory: base_output_dir/images/
            images_dir = os.path.join(base_output_dir, 'images')

        issues_found = 0

        print(f"Verifying dimensions for {len(coco_data['images'])} images...")
        if split_dir:
            if '_annotations.coco.json' in coco_file_path:
                print(f"  Using Coconuts-1 style split directory: {split_dir}/")
            else:
                print(f"  Using split directory: {split_dir}/images/")

        for img_info in coco_data['images']:
            img_path = os.path.join(images_dir, img_info['file_name'])

            if not os.path.exists(img_path):
                print(f"⚠ Missing image file: {img_info['file_name']}")
                issues_found += 1
                continue

            # Read actual image dimensions
            actual_image = cv2.imread(img_path)
            if actual_image is None:
                print(f"⚠ Cannot read image: {img_info['file_name']}")
                issues_found += 1
                continue

            actual_height, actual_width = actual_image.shape[:2]
            expected_width = img_info['width']
            expected_height = img_info['height']

            if actual_width != expected_width or actual_height != expected_height:
                print(f"⚠ Dimension mismatch in {img_info['file_name']}:")
                print(f"    Expected: {expected_width}x{expected_height}")
                print(f"    Actual: {actual_width}x{actual_height}")
                issues_found += 1

        if issues_found == 0:
            print("✓ All image dimensions verified correctly")

        return issues_found

    except Exception as e:
        print(f"Error during dimension verification: {e}")
        return -1


def create_shapefile_visualization(image, shapefile, transform, output_path, title="Shapefile Visualization"):
    """
    Create visualization showing shapefile polygons overlaid on image.
    Uses shapefile row index for colors (like clean_mask.py), not category_id.
    This ensures maximum color distinction regardless of which counties are present.

    Args:
        image: RGB image array
        shapefile: GeoDataFrame with county polygons (ALREADY MASKED if masks were applied)
        transform: Rasterio affine transform for geo→pixel coordinate conversion
        output_path: Where to save visualization PNG
        title: Title for the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    # Generate colors for each shapefile row (not category_id!)
    # This matches clean_mask.py behavior
    num_counties = len(shapefile)
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_counties))

    # Draw each county polygon
    for idx, row in shapefile.iterrows():
        geom = row.geometry

        # Handle both Polygon and MultiPolygon
        if geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)
        elif geom.geom_type == 'Polygon':
            polygons = [geom]
        else:
            continue

        color = colors[idx % len(colors)]

        # Draw all polygons for this county
        for poly in polygons:
            exterior = poly.exterior.coords[:]
            pixel_coords = [~transform * (x, y) for x, y in exterior]
            xs, ys = zip(*pixel_coords)

            # Fill and outline with same color
            ax.fill(xs, ys, alpha=0.3, fc=color, ec=color, linewidth=2)
            ax.plot(xs, ys, color=color, linewidth=2)

            # Draw holes (if any)
            for interior in poly.interiors:
                hole_coords = interior.coords[:]
                hole_pixel_coords = [~transform * (x, y) for x, y in hole_coords]
                hole_xs, hole_ys = zip(*hole_pixel_coords)
                ax.plot(hole_xs, hole_ys, color=color, linewidth=1, linestyle='-', alpha=0.5)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    Shapefile visualization saved: {os.path.basename(output_path)}")


def create_hierarchical_visualization(
    image,
    annotations_l0,
    annotations_l1,
    categories_l0,
    categories_l1,
    output_path,
    title="Hierarchical Segmentation"
):
    """
    Create visualization showing both county (L0) and township (L1) annotations.

    This creates a 2-panel figure:
    - Left: County level with colored fills and thick boundaries
    - Right: Township level with distinct colors and clear boundaries

    Args:
        image: RGB image array
        annotations_l0: List of COCO annotations for county level
        annotations_l1: List of COCO annotations for township level
        categories_l0: List of county categories (with 'english' field)
        categories_l1: List of township categories (with 'english' field)
        output_path: Path to save the visualization
        title: Title for the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

    # Setup colormaps - use tab20 for counties
    colors_l0 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))

    # Panel 1: County level (L0)
    ax1.imshow(image)
    ax1.set_title(f"L0: County Level ({len(annotations_l0)} regions)", fontsize=16, fontweight='bold', color='#2E86AB')

    for ann in annotations_l0:
        category_id = ann['category_id']
        segmentation = ann['segmentation'][0] if ann['segmentation'] else []

        if len(segmentation) < 6:
            continue

        points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
        color = colors_l0[category_id % len(colors_l0)]

        # Filled polygon with thick white-outlined boundary
        polygon_patch = Polygon(
            points,
            facecolor=color,
            edgecolor='white',
            alpha=0.6,
            linewidth=3
        )
        ax1.add_patch(polygon_patch)

        # Add darker boundary on top for definition
        xs = [p[0] for p in points] + [points[0][0]]
        ys = [p[1] for p in points] + [points[0][1]]
        ax1.plot(xs, ys, color=tuple(c * 0.7 for c in color[:3]), linewidth=2, zorder=5)

        # Add label at centroid - USE ENGLISH NAME
        if points:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)

            # Get English name, fallback to ID
            cat_info = next((c for c in categories_l0 if c['id'] == category_id), None)
            if cat_info:
                label = cat_info.get('english', cat_info.get('name', f"C{category_id}"))
            else:
                label = f"C{category_id}"
            short_name = label[:12] if len(label) > 12 else label

            ax1.text(cx, cy, short_name, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=tuple(c * 0.8 for c in color[:3]),
                             edgecolor='white', linewidth=2, alpha=0.95),
                    zorder=10)

    ax1.set_xlim(0, image.shape[1])
    ax1.set_ylim(image.shape[0], 0)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Panel 2: Township level (L1) - USE DISTINCT COLORS PER TOWNSHIP
    ax2.imshow(image)
    ax2.set_title(f"L1: Township Level ({len(annotations_l1)} regions)", fontsize=16, fontweight='bold', color='#A23B72')

    # Use a larger colormap for townships - each gets distinct color
    n_townships = len(annotations_l1)
    if n_townships <= 20:
        colors_l1 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    else:
        # Use rainbow/spectral for more distinction
        colors_l1 = cm.get_cmap('nipy_spectral')(np.linspace(0.1, 0.9, n_townships))

    for idx, ann in enumerate(annotations_l1):
        category_id = ann['category_id']
        segmentation = ann['segmentation'][0] if ann['segmentation'] else []

        if len(segmentation) < 6:
            continue

        points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]

        # Each township gets its own distinct color
        color = colors_l1[idx % len(colors_l1)]

        # Filled polygon with BLACK boundary for clear separation
        polygon_patch = Polygon(
            points,
            facecolor=color,
            edgecolor='black',
            alpha=0.7,
            linewidth=2
        )
        ax2.add_patch(polygon_patch)

        # Add labels - USE ENGLISH NAME
        if points:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)

            # Get English name
            cat_info = next((c for c in categories_l1 if c['id'] == category_id), None)
            if cat_info:
                label = cat_info.get('english', f"T{category_id}")
            else:
                label = f"T{category_id}"
            short_name = label[:8] if len(label) > 8 else label

            # Show labels for all townships (they're important)
            ax2.text(cx, cy, short_name, ha='center', va='center',
                    fontsize=7, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                             edgecolor='black', linewidth=1, alpha=0.85))

    ax2.set_xlim(0, image.shape[1])
    ax2.set_ylim(image.shape[0], 0)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Add overall title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=220, bbox_inches='tight', facecolor='white')
        print(f"    Hierarchical visualization saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"    Error saving hierarchical visualization: {e}")

    plt.close(fig)


def _create_polygon_with_holes(segmentation_list, facecolor, edgecolor, alpha, linewidth):
    """
    Create a matplotlib patch that properly handles polygons with holes.

    Args:
        segmentation_list: List of coordinate lists. First is exterior, rest are holes.
        facecolor: Fill color
        edgecolor: Edge color
        alpha: Transparency
        linewidth: Edge width

    Returns:
        PathPatch object or None if invalid
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    if not segmentation_list or len(segmentation_list[0]) < 6:
        return None

    # Build path vertices and codes
    all_vertices = []
    all_codes = []

    for ring_idx, ring in enumerate(segmentation_list):
        if len(ring) < 6:
            continue

        # Convert flat coords to points
        points = [(ring[i], ring[i+1]) for i in range(0, len(ring), 2)]

        # Add to path
        for i, pt in enumerate(points):
            all_vertices.append(pt)
            if i == 0:
                all_codes.append(Path.MOVETO)
            else:
                all_codes.append(Path.LINETO)

        # Close the ring
        all_vertices.append(points[0])
        all_codes.append(Path.CLOSEPOLY)

    if not all_vertices:
        return None

    path = Path(all_vertices, all_codes)
    return PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                     alpha=alpha, linewidth=linewidth)


def create_hierarchical_overlay_visualization(
    image,
    annotations_l0,
    annotations_l1,
    categories_l0,
    categories_l1,
    output_path,
    title="Hierarchical Overlay"
):
    """
    Create single-panel visualization with both levels overlaid.

    County (L0): thick colored boundaries with white outline for contrast
    Township (L1): distinct colored fills with black boundaries

    Uses ENGLISH labels only to avoid font issues.
    Properly handles polygons with holes (from mask subtraction).

    Args:
        image: RGB image array
        annotations_l0: County level annotations
        annotations_l1: Township level annotations
        categories_l0: County categories (with 'english' field)
        categories_l1: Township categories (with 'english' field)
        output_path: Path to save
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)

    n_l1 = len(annotations_l1) if annotations_l1 else 0

    # Use tab20 for counties
    colors_l0 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))

    # Use spectral colormap for townships - each gets distinct color
    if n_l1 <= 20:
        colors_l1 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
    else:
        colors_l1 = cm.get_cmap('nipy_spectral')(np.linspace(0.1, 0.9, n_l1))

    # Draw L1 (townships) first - underneath with distinct fills
    for idx, ann in enumerate(annotations_l1):
        category_id = ann['category_id']
        segmentation = ann.get('segmentation', [])

        if not segmentation or len(segmentation[0]) < 6:
            continue

        # Each township gets distinct color
        fill_color = colors_l1[idx % len(colors_l1)]

        # Use PathPatch for polygons with holes
        patch = _create_polygon_with_holes(
            segmentation,
            facecolor=fill_color[:3],
            edgecolor='black',
            alpha=0.5,
            linewidth=1.5
        )
        if patch:
            ax.add_patch(patch)

        # Add township label - USE ENGLISH (use exterior centroid)
        ext_coords = segmentation[0]
        points = [(ext_coords[i], ext_coords[i+1]) for i in range(0, len(ext_coords), 2)]
        if points:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)

            cat_info = next((c for c in categories_l1 if c['id'] == category_id), None)
            if cat_info:
                label = cat_info.get('english', f"T{category_id}")
            else:
                label = f"T{category_id}"
            short_name = label[:6] if len(label) > 6 else label

            ax.text(cx, cy, short_name, ha='center', va='center',
                    fontsize=6, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.15", facecolor='white', alpha=0.7),
                    zorder=5)

    # Draw L0 (counties) on top - thick boundaries with white outline for contrast
    for ann in annotations_l0:
        category_id = ann['category_id']
        segmentation = ann.get('segmentation', [])

        if not segmentation or len(segmentation[0]) < 6:
            continue

        # Draw exterior boundary
        ext_coords = segmentation[0]
        points = [(ext_coords[i], ext_coords[i+1]) for i in range(0, len(ext_coords), 2)]
        color = colors_l0[category_id % len(colors_l0)]

        xs = [p[0] for p in points] + [points[0][0]]
        ys = [p[1] for p in points] + [points[0][1]]

        # Draw white outline first (for contrast)
        ax.plot(xs, ys, color='white', linewidth=7, solid_capstyle='round', zorder=10)
        # Draw colored boundary on top
        ax.plot(xs, ys, color=color, linewidth=4, solid_capstyle='round', zorder=11)

        # Also draw hole boundaries if any
        for hole_coords in segmentation[1:]:
            if len(hole_coords) >= 6:
                hole_points = [(hole_coords[i], hole_coords[i+1]) for i in range(0, len(hole_coords), 2)]
                hxs = [p[0] for p in hole_points] + [hole_points[0][0]]
                hys = [p[1] for p in hole_points] + [hole_points[0][1]]
                # Draw hole boundary with same style but dashed
                ax.plot(hxs, hys, color='white', linewidth=5, solid_capstyle='round', zorder=10)
                ax.plot(hxs, hys, color=color, linewidth=3, linestyle='--', solid_capstyle='round', zorder=11)

        # Add county label - USE ENGLISH
        if points:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)

            cat_info = next((c for c in categories_l0 if c['id'] == category_id), None)
            if cat_info:
                label = cat_info.get('english', f"C{category_id}")
            else:
                label = f"C{category_id}"
            short_name = label[:12] if len(label) > 12 else label

            ax.text(cx, cy, short_name, ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                             edgecolor='white', linewidth=2, alpha=0.95),
                    zorder=12)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title with counts
    ax.set_title(f"{title}\nCounty (L0): {len(annotations_l0)} | Township (L1): {len(annotations_l1)}",
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=250, bbox_inches='tight', facecolor='white')
        print(f"    Hierarchical overlay saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"    Error saving hierarchical overlay: {e}")

    plt.close(fig)
