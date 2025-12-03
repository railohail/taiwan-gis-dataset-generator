"""
GIS geometry and shapefile utilities.

This module provides:
- TIF file discovery and loading
- Shapefile loading with CRS reprojection
- District/county image extraction
- Window generation for sliding-window crops
- Polygon clipping and coordinate transformations
- Mask region application to shapefiles
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box, Polygon as ShapelyPolygon
from shapely.ops import transform
import matplotlib.pyplot as plt
import os
import glob


def find_tif_files(district_folder, mapdata_base_dir='datasets/MAPDATA'):
    """Find all TIF files in a district folder."""
    # Northern Taiwan districts processing
    northern_districts = ['taipei', 'new_taipei', 'keelung', 'taoyuan', 'hsinchu', 'yilan']

    if district_folder in northern_districts:
        # Look for TIF files in the district folder
        tif_pattern = os.path.join(mapdata_base_dir, district_folder, '*.tif')
        tif_files = glob.glob(tif_pattern)

        # Filter out auxiliary files
        tif_files = [f for f in tif_files if not f.endswith('.aux.xml')]

        return sorted(tif_files) if tif_files else []

    # Default behavior for other districts
    tif_pattern = os.path.join(mapdata_base_dir, district_folder, '*.tif')
    tif_files = glob.glob(tif_pattern)

    # Filter out auxiliary files
    tif_files = [f for f in tif_files if not f.endswith('.aux.xml')]

    return sorted(tif_files)


def load_data(raster_path, shapefile_path):
    """Load raster and shapefile, reproject shapefile to raster CRS."""
    try:
        raster = rasterio.open(raster_path)
    except rasterio.RasterioIOError as e:
        print(f"Error opening raster file {raster_path}: {e}")
        return None, None

    try:
        shapefile = gpd.read_file(shapefile_path)
    except Exception as e:
        print(f"Error opening shapefile {shapefile_path}: {e}")
        if raster:
            raster.close()
        return None, None

    # Reproject shapefile to raster CRS if necessary
    if shapefile.crs != raster.crs:
        shapefile = shapefile[shapefile.geometry.is_valid]
        if shapefile.empty:
            print("Error: Shapefile contains no valid geometries after filtering.")
            raster.close()
            return None, None
        try:
            shapefile = shapefile.to_crs(raster.crs)
        except Exception as e:
            print(f"Error reprojecting shapefile: {e}")
            raster.close()
            return None, None

    return raster, shapefile


def extract_single_district_image(raster, shapefile, county_name, buffer_pixels=100, crop_factor=0.0):
    """
    Extract image containing only the specified district with buffer and mask out other areas.

    Args:
        raster: Rasterio dataset
        shapefile: GeoDataFrame with county polygons
        county_name: Name of county to extract
        buffer_pixels: Additional pixels to add around county bounds (default: 100)
        crop_factor: Proportion to crop from edges (0.0-1.0, default: 0.0 = no crop)
                     Example: 0.05 = crop 5% from each edge
    """
    # Filter shapefile for the specific county
    county_gdf = shapefile[shapefile.apply(
        lambda row: any(
            str(row[col]).strip() == county_name
            for col in ['COUNTYNAME', 'NAME', 'County', 'county', '縣市名稱', 'COUNTY', 'COUNTYENG', 'C_Name']
            if col in shapefile.columns and row[col] is not None
        ), axis=1
    )]

    if county_gdf.empty:
        print(f"    No shapefile data found for county: {county_name}")
        return None, None, None, None

    # Get bounds of the county geometry
    bounds = county_gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Convert geographic bounds to pixel coordinates
    transform = raster.transform

    # Get pixel coordinates for bounds
    left_px, top_px = ~transform * (bounds[0], bounds[3])
    right_px, bottom_px = ~transform * (bounds[2], bounds[1])

    # Ensure proper ordering and add buffer
    min_x = max(0, int(min(left_px, right_px)) - buffer_pixels)
    max_x = min(raster.width, int(max(left_px, right_px)) + buffer_pixels)
    min_y = max(0, int(min(top_px, bottom_px)) - buffer_pixels)
    max_y = min(raster.height, int(max(top_px, bottom_px)) + buffer_pixels)

    # Apply crop_factor if specified (crop from all edges like combined mode)
    if crop_factor > 0:
        width = max_x - min_x
        height = max_y - min_y
        crop_x = int(width * crop_factor)
        crop_y = int(height * crop_factor)

        # Apply crop while staying within bounds
        min_x = max(0, min_x + crop_x)
        max_x = min(raster.width, max_x - crop_x)
        min_y = max(0, min_y + crop_y)
        max_y = min(raster.height, max_y - crop_y)

    # Check if bounds are valid
    if min_x >= max_x or min_y >= max_y:
        return None, None, None, None

    # Read the cropped region from raster
    raster_data = raster.read(window=rasterio.windows.Window(min_x, min_y, max_x - min_x, max_y - min_y))

    # Calculate new transform for the cropped region
    new_transform = rasterio.windows.transform(
        rasterio.windows.Window(min_x, min_y, max_x - min_x, max_y - min_y),
        transform
    )

    # Convert raster data to image format
    if len(raster_data.shape) == 3:
        image_data = np.transpose(raster_data, (1, 2, 0))
    else:
        image_data = raster_data

    # Create a mask for the specific county
    county_mask = features.rasterize(
        shapes=[(geom, 1) for geom in county_gdf.geometry],
        out_shape=(max_y - min_y, max_x - min_x),
        transform=new_transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    )

    if np.sum(county_mask) == 0:
        return None, None, None, None

    # Apply mask to image - set areas outside the county to a dark gray instead of pure black
    if len(image_data.shape) == 3:
        # RGB image
        masked_image_data = image_data.copy().astype(np.float32)

        # Create 3-channel mask
        mask_3d = np.stack([county_mask, county_mask, county_mask], axis=2)

        # Apply mask: keep original values inside mask, set to dark gray (64) outside
        masked_image_data = masked_image_data * mask_3d + 64 * (1 - mask_3d)
        masked_image_data = masked_image_data.astype(np.uint8)
    else:
        # Grayscale or single band
        masked_image_data = image_data.astype(np.float32)
        masked_image_data = masked_image_data * county_mask + 64 * (1 - county_mask)
        masked_image_data = masked_image_data.astype(np.uint8)

    # Convert back to raster format
    if len(masked_image_data.shape) == 3:
        masked_raster_data = np.transpose(masked_image_data, (2, 0, 1))
    else:
        masked_raster_data = masked_image_data

    # Debug: Save mask and preview for debugging
    debug_dir = "debug_masks"
    os.makedirs(debug_dir, exist_ok=True)

    safe_county_name = county_name.replace(' ', '_').replace('市', 'City').replace('縣', 'County')

    # Save mask as image
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(county_mask, cmap='gray')
    plt.title(f'Mask for {safe_county_name}')
    plt.axis('off')

    # Save masked result
    plt.subplot(1, 2, 2)
    if len(masked_image_data.shape) == 3:
        plt.imshow(masked_image_data)
    else:
        plt.imshow(masked_image_data, cmap='gray')
    plt.title(f'Masked Image for {safe_county_name}')
    plt.axis('off')

    debug_path = os.path.join(debug_dir, f'debug_{safe_county_name}.png')
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate geographic bounds of the extracted district image
    # This is needed for mask coverage calculations in separate mode
    district_bounds_geo = rasterio.transform.array_bounds(
        max_y - min_y, max_x - min_x, new_transform
    )

    # CRITICAL FIX: Clip the shapefile to the actual extracted image bounds
    # This ensures annotations only cover the portion of the county visible in the cropped image
    bbox = box(*district_bounds_geo)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs=county_gdf.crs)
    clipped_shapefile = gpd.clip(county_gdf, bbox_gdf)

    # Verify clipping worked
    if clipped_shapefile.empty:
        print(f"    WARNING: Clipping resulted in empty shapefile for {county_name}")
        # Fall back to original if clipping failed
        clipped_shapefile = county_gdf.copy()

    return masked_raster_data, new_transform, district_bounds_geo, clipped_shapefile


def crop_image_and_shapefile(raster, image_data, shapefile, crop_factor):
    """Crop the raster image and clip the shapefile to the new bounds."""
    height, width = image_data.shape[:2]
    crop_x = int(width * crop_factor)
    crop_y = int(height * crop_factor)

    if (2 * crop_x >= width) or (2 * crop_y >= height):
        print("Error: Crop factor is too large.")
        return None, None, None, None

    # Crop the image
    cropped_image = image_data[crop_y:height-crop_y, crop_x:width-crop_x]

    # Update the geotransform
    transform = raster.transform
    cropped_transform = rasterio.Affine(
        transform.a, transform.b,
        transform.c + transform.a * crop_x + transform.b * crop_y,
        transform.d, transform.e,
        transform.f + transform.d * crop_x + transform.e * crop_y
    )

    # Get bounds and clip shapefile
    cropped_bounds = rasterio.transform.array_bounds(
        cropped_image.shape[0], cropped_image.shape[1], cropped_transform
    )

    bbox = box(*cropped_bounds)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs=raster.crs)
    clipped_shapefile = gpd.clip(shapefile, bbox_gdf)

    return cropped_image, cropped_transform, cropped_bounds, clipped_shapefile


def generate_windows(image, window_config):
    """Generate 4 overlapping windows from the full image."""
    height, width = image.shape[:2]

    window_width = int(width * window_config['x_percent'] / 100)
    window_height = int(height * window_config['y_percent'] / 100)

    step_x = (width - window_width) // 1
    step_y = (height - window_height) // 1

    windows = []
    window_info = []

    for row in range(2):
        for col in range(2):
            start_x = col * step_x
            start_y = row * step_y

            end_x = min(start_x + window_width, width)
            end_y = min(start_y + window_height, height)

            window = image[start_y:end_y, start_x:end_x]
            windows.append(window)

            window_info.append({
                'id': row * 2 + col,
                'start_x': start_x,
                'start_y': start_y,
                'end_x': end_x,
                'end_y': end_y,
                'width': end_x - start_x,
                'height': end_y - start_y
            })

    return windows, window_info


def clip_polygon_to_window(polygon, start_x, start_y, end_x, end_y):
    """Clip a polygon to a window boundary and adjust coordinates.

    Returns:
        A Polygon, MultiPolygon, or None if clipping results in nothing
    """
    window_bounds = box(start_x, start_y, end_x, end_y)
    clipped_geom = polygon.intersection(window_bounds)

    if clipped_geom.is_empty:
        return None

    # Don't reduce MultiPolygon to single polygon - keep all parts!
    # The caller will handle iterating through multiple parts
    if clipped_geom.geom_type not in ['Polygon', 'MultiPolygon']:
        return None

    def adjust_coords(x, y, z=None):
        return x - start_x, y - start_y

    adjusted_geom = transform(adjust_coords, clipped_geom)
    return adjusted_geom


def calculate_mask_coverage_percentage(shapefile_gdf, mask_regions, raster_transform):
    """
    Calculate what percentage of the shapefile area is covered by mask regions.

    Args:
        shapefile_gdf: GeoDataFrame with county polygons
        mask_regions: List of mask region dicts with keys: x, y, width, height (pixels)
        raster_transform: Rasterio Affine transform for pixel→geo coordinate conversion

    Returns:
        float: Percentage of total area covered by masks (0.0 - 100.0)
    """
    from shapely.ops import unary_union
    from shapely.geometry import box

    if not mask_regions or len(mask_regions) == 0:
        return 0.0

    # Calculate total area of all county polygons
    total_area = shapefile_gdf.geometry.area.sum()
    if total_area == 0:
        return 0.0

    # Convert mask rectangles to geographic coordinates
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

    # Union all mask rectangles
    combined_mask = unary_union(mask_polygons)

    # Calculate intersection area (how much of counties is covered by masks)
    union_geometry = unary_union(shapefile_gdf.geometry)
    intersection_area = union_geometry.intersection(combined_mask).area

    # Return percentage
    coverage_pct = (intersection_area / total_area) * 100.0
    return coverage_pct


def apply_mask_regions_to_shapefile(shapefile_gdf, mask_regions, raster_transform,
                                     return_coverage=False, skip_threshold=None, image_bounds=None):
    """
    Apply mask regions to shapefile by subtracting mask rectangles from polygons.

    This creates holes in county polygons where zoom-ins, legends, or other
    decorative elements exist, preventing annotations in those areas.

    IMPORTANT: If image_bounds is provided, the function will first expand the
    shapefile polygon to cover the full image bounds, then subtract masks.
    This ensures masks in image margins (outside polygon boundary) are still excluded.

    Args:
        shapefile_gdf: GeoDataFrame with county polygons
        mask_regions: List of mask region dicts with keys: x, y, width, height (pixels)
        raster_transform: Rasterio Affine transform for pixel→geo coordinate conversion
        return_coverage: If True, return (shapefile, coverage_pct) tuple
        skip_threshold: If set and coverage exceeds this %, return None to signal skip
                       (e.g., 50.0 means skip if >50% is masked)
        image_bounds: Optional tuple (minx, miny, maxx, maxy) in geographic coords.
                     If provided, polygon is clipped to image bounds AFTER mask subtraction.

    Returns:
        GeoDataFrame with masked polygons (holes cut out where masks were applied)
        OR (GeoDataFrame, coverage_pct) tuple if return_coverage=True
        OR None if skip_threshold exceeded
        Returns original shapefile if mask_regions is None or empty
    """
    from shapely.ops import unary_union

    # Return original if no masks
    if not mask_regions or len(mask_regions) == 0:
        if return_coverage:
            return shapefile_gdf, 0.0
        return shapefile_gdf

    # Convert mask rectangles from pixel coordinates to geographic coordinates
    mask_polygons = []
    for region in mask_regions:
        x1_px, y1_px = region['x'], region['y']
        x2_px, y2_px = x1_px + region['width'], y1_px + region['height']

        # Convert pixel corners to geographic coordinates
        x1_geo, y1_geo = raster_transform * (x1_px, y1_px)
        x2_geo, y2_geo = raster_transform * (x2_px, y2_px)

        # Create polygon in geographic coordinates
        mask_poly = box(
            min(x1_geo, x2_geo),
            min(y1_geo, y2_geo),
            max(x1_geo, x2_geo),
            max(y1_geo, y2_geo)
        )
        mask_polygons.append(mask_poly)

    # Union all mask rectangles into one geometry
    combined_mask = unary_union(mask_polygons)

    # Calculate coverage percentage if needed for skip_threshold check
    coverage_pct = None
    if skip_threshold is not None or return_coverage:
        total_area = shapefile_gdf.geometry.area.sum()
        if total_area > 0:
            union_geometry = unary_union(shapefile_gdf.geometry)
            intersection_area = union_geometry.intersection(combined_mask).area
            coverage_pct = (intersection_area / total_area) * 100.0

            # Debug output
            if skip_threshold is not None:
                print(f"    DEBUG: District area = {total_area:.2f}, Mask intersection = {intersection_area:.2f}, Coverage = {coverage_pct:.1f}%")
        else:
            coverage_pct = 0.0

    # Check skip threshold
    if skip_threshold is not None and coverage_pct is not None:
        if coverage_pct > skip_threshold:
            print(f"    Masking: SKIPPED - {coverage_pct:.1f}% of area is masked (threshold: {skip_threshold}%)")
            return None

    # Subtract mask from each county polygon
    masked_shapefile = shapefile_gdf.copy()

    # If image_bounds provided, use union approach to ensure masks work everywhere
    if image_bounds is not None:
        # NEW APPROACH: Union polygon with image bbox, subtract masks, then intersect back
        # This ensures masks cut out even if they're outside the original polygon
        image_bbox = box(*image_bounds)

        masked_shapefile['geometry'] = masked_shapefile['geometry'].apply(
            lambda geom: geom.union(image_bbox).difference(combined_mask).intersection(image_bbox)
        )
    else:
        # Original behavior: only subtract where mask intersects polygon
        masked_shapefile['geometry'] = masked_shapefile['geometry'].apply(
            lambda geom: geom.difference(combined_mask) if geom.intersects(combined_mask) else geom
        )

    # Remove empty geometries (if mask completely covers a county)
    masked_shapefile = masked_shapefile[~masked_shapefile.geometry.is_empty]

    # Log statistics
    original_count = len(shapefile_gdf)
    masked_count = len(masked_shapefile)
    removed_count = original_count - masked_count

    if removed_count > 0:
        if coverage_pct is not None:
            print(f"    Masking: {original_count} → {masked_count} polygons ({removed_count} completely masked, {coverage_pct:.1f}% coverage)")
        else:
            print(f"    Masking: {original_count} → {masked_count} polygons ({removed_count} completely masked)")
    else:
        if coverage_pct is not None:
            print(f"    Masking: Applied {len(mask_regions)} mask region(s) to {masked_count} polygon(s) ({coverage_pct:.1f}% coverage)")
        else:
            print(f"    Masking: Applied {len(mask_regions)} mask region(s) to {masked_count} polygon(s)")

    # Return with or without coverage
    if return_coverage:
        return masked_shapefile, coverage_pct
    return masked_shapefile
