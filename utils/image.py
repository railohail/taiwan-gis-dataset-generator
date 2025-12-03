"""
Image processing and augmentation utilities.

This module provides:
- Grayscale conversion
- Unicode-safe image writing (Windows compatibility)
- Noise generation (Gaussian, salt-pepper, speckle, Perlin, textured)
- Distance-based noise application
- HSV color augmentation
- Image rotation with annotation transformation
- Image scaling with annotation adjustment
"""

import numpy as np
import cv2
import os
import random
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from rasterio import features
from skimage import color
from shapely.geometry import box, Polygon as ShapelyPolygon
import rasterio


def convert_to_grayscale(image):
    """
    Convert RGB image to grayscale (3-channel for compatibility).

    Args:
        image: RGB image (numpy array, shape HxWx3)

    Returns:
        Grayscale image as 3-channel (HxWx3) for compatibility
    """
    if len(image.shape) == 2:
        # Already grayscale, convert to 3-channel
        return np.stack([image, image, image], axis=2)

    if image.shape[2] == 1:
        # Single channel, convert to 3-channel
        gray = image[:, :, 0]
        return np.stack([gray, gray, gray], axis=2)

    # Convert RGB to grayscale using standard luminance formula
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert back to 3-channel for compatibility with existing pipeline
    return np.stack([gray, gray, gray], axis=2)


def unicode_safe_imwrite(filepath, img, verify=False):
    """
    Unicode-safe version of cv2.imwrite for Windows compatibility.
    Uses cv2.imencode + file writing to handle Unicode filenames.

    Args:
        filepath: Path to save image
        img: Image data (numpy array)
        verify: If True, verify image was saved correctly (slower, default False)

    Returns:
        (success, width, height) tuple
    """
    height, width = img.shape[:2]

    try:
        # First try standard cv2.imwrite
        success = cv2.imwrite(filepath, img)
        if success:
            return (True, width, height)
    except (cv2.error, OSError, IOError):
        # cv2.imwrite failed (likely due to Unicode path on Windows)
        # Will try fallback method below
        pass

    # Fallback: use cv2.imencode + file writing for Unicode compatibility
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            ext = '.png'  # Default to PNG

        success, encoded_img = cv2.imencode(ext, img)
        if success:
            with open(filepath, 'wb') as f:
                f.write(encoded_img.tobytes())

            # Optional verification (slower)
            if verify:
                saved_img = cv2.imread(filepath)
                if saved_img is not None:
                    actual_height, actual_width = saved_img.shape[:2]
                    return (True, actual_width, actual_height)

            return (True, width, height)
    except Exception as e:
        print(f"    ERROR: Failed to save image {filepath}: {e}")
        return (False, 0, 0)

    return (False, 0, 0)


def apply_hue_augmentation(image, hue_shift=0.0, saturation_factor=1.0, value_factor=1.0):
    """Apply hue, saturation, and value augmentation to RGB image."""
    # Convert RGB to HSV
    hsv_image = color.rgb2hsv(image)

    # Apply hue shift (wrap around at 0 and 1)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 1.0

    # Apply saturation scaling (clamp to [0, 1])
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 1)

    # Apply value/brightness scaling (clamp to [0, 1])
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * value_factor, 0, 1)

    # Convert back to RGB and scale to uint8
    augmented_rgb = color.hsv2rgb(hsv_image)
    augmented_uint8 = (augmented_rgb * 255).astype(np.uint8)

    return augmented_uint8


def generate_hue_augmentation_params(count, hue_range, sat_range, val_range):
    """Generate random parameters for hue augmentation."""
    params = []
    for _ in range(count):
        hue_shift = random.uniform(hue_range[0], hue_range[1])
        sat_factor = random.uniform(sat_range[0], sat_range[1])
        val_factor = random.uniform(val_range[0], val_range[1])
        params.append((hue_shift, sat_factor, val_factor))
    return params


def preprocess_raster(raster_data):
    """Convert raster data to uint8 RGB format."""
    full_rgb = np.transpose(raster_data, (1, 2, 0))

    # Normalize to 0-255 range
    if full_rgb.max() > 1.0 and full_rgb.max() <= 255:
        full_rgb = full_rgb.astype(np.uint8)
    elif full_rgb.max() > 255:
        print("Normalizing raster data from > 255 range.")
        full_rgb = (full_rgb / full_rgb.max() * 255).astype(np.uint8)
    elif full_rgb.max() <= 1.0:
        print("Normalizing raster data from 0-1 range.")
        full_rgb = (full_rgb * 255).astype(np.uint8)
    else:
        print(f"Warning: Unexpected raster data range (max value: {full_rgb.max()}). Assuming it's already scaled.")
        full_rgb = full_rgb.astype(np.uint8)

    # Handle different band configurations
    if full_rgb.shape[2] == 3:  # RGB
        return full_rgb
    elif full_rgb.shape[2] >= 4:  # RGBA or more bands
        print("Using first 3 bands for RGB.")
        return full_rgb[:,:,:3]
    elif full_rgb.shape[2] == 1:  # Grayscale
        grayscale = full_rgb[:,:,0]
        return np.stack([grayscale, grayscale, grayscale], axis=2)
    else:
        print(f"Error: Unexpected number of bands in raster: {full_rgb.shape[2]}")
        return None


def generate_noise(shape, noise_type, intensity):
    """Generate different types of noise based on specified parameters."""
    if noise_type == 'gaussian':
        return np.random.normal(0, intensity * 255, shape).astype(np.int16)

    elif noise_type == 'salt_pepper':
        noise = np.zeros(shape, dtype=np.int16)
        salt_prob = intensity * 0.5
        salt_mask = np.random.random(shape) < salt_prob
        noise[salt_mask] = 255
        pepper_prob = intensity * 0.5
        pepper_mask = np.random.random(shape) < pepper_prob
        noise[pepper_mask] = -255
        return noise

    elif noise_type == 'speckle':
        base = np.ones(shape, dtype=np.int16)
        return (base * np.random.randn(*shape) * intensity * 255).astype(np.int16)

    elif noise_type == 'perlin':
        base_noise = np.random.rand(*shape)
        smoothed = ndimage.gaussian_filter(base_noise, sigma=2.0)
        return ((smoothed - 0.5) * 2 * intensity * 255).astype(np.int16)

    elif noise_type == 'textured':
        h, w = shape[:2] if len(shape) > 2 else shape
        y = np.linspace(0, h, h)
        x = np.linspace(0, w, w)
        X, Y = np.meshgrid(x, y)

        high_freq = np.sin(X/2) * np.cos(Y/2)
        med_freq = np.sin(X/5) * np.cos(Y/5)
        low_freq = np.sin(X/20) * np.cos(Y/20)

        combined = (high_freq * 0.5 + med_freq * 0.3 + low_freq * 0.2)
        combined = (combined - combined.min()) / (combined.max() - combined.min())
        combined = ((combined - 0.5) * 2 * intensity * 255).astype(np.int16)

        if len(shape) > 2:
            channels = shape[2]
            return np.stack([combined] * channels, axis=2)
        return combined

    else:
        return np.random.normal(0, intensity * 255, shape).astype(np.int16)


def apply_distance_based_noise(image, shapefile, raster_transform, noise_config, verbose=False):
    """Apply noise to image with distance-based intensity, preserving borders."""
    if verbose:
        print(f"  Applying {noise_config['name']} noise (intensity: {noise_config['intensity']})...")

    img_shape = image.shape[:2]

    # Create buffer around county boundaries
    buffer_distance_pixels = noise_config['border_buffer_pixels']

    # Convert buffer from pixels to geographic units
    pixel_size = abs(raster_transform.a)
    buffer_distance_geo = buffer_distance_pixels * pixel_size

    # Create inner buffer to define the noisy region
    buffered_inner = shapefile.copy()
    buffered_inner['geometry'] = buffered_inner.geometry.buffer(-buffer_distance_geo)

    # Remove invalid or zero-area geometries
    buffered_inner = buffered_inner[buffered_inner.geometry.is_valid & (buffered_inner.geometry.area > 0)]

    if buffered_inner.empty:
        if verbose:
            print(f"    Warning: No valid geometries remain after buffering.")
        noisy_region_mask = np.zeros(img_shape, dtype=np.uint8)
    else:
        # Rasterize the buffered polygons
        inner_shapes = ((geom, 1) for geom in buffered_inner.geometry)
        noisy_region_mask = features.rasterize(
            shapes=inner_shapes,
            out_shape=img_shape,
            transform=raster_transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

    # Calculate distance-based noise intensity per polygon
    final_noise_intensity_map = np.zeros(img_shape, dtype=float)

    for index, row in shapefile.iterrows():
        geometry = row.geometry

        # Handle MultiPolygon case
        if geometry.geom_type == 'MultiPolygon':
            geometry = max(geometry.geoms, key=lambda x: x.area)

        if geometry.geom_type != 'Polygon':
            continue

        # Rasterize the single polygon
        single_boundary_mask = features.rasterize(
            shapes=[(geometry, 1)],
            out_shape=img_shape,
            transform=raster_transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

        if np.sum(single_boundary_mask) == 0:
            continue

        # Calculate distance transform
        single_distance_map = distance_transform_edt(single_boundary_mask)

        # Normalize distance
        max_distance = np.max(single_distance_map)
        if max_distance == 0:
            continue

        normalized_single_distance = np.zeros_like(single_distance_map, dtype=float)
        mask_pixels = (single_boundary_mask == 1)
        normalized_single_distance[mask_pixels] = single_distance_map[mask_pixels] / max_distance

        # Calculate noise intensity using exponential function
        acceleration = noise_config['acceleration']
        sigma = 1 / (2 * acceleration + 1e-6)
        single_intensity_map_raw = np.exp((normalized_single_distance - 1) / sigma)
        single_intensity_map = single_intensity_map_raw * noise_config['intensity']

        # Combine with the final map
        final_noise_intensity_map = np.maximum(final_noise_intensity_map, single_intensity_map)

    # Generate base noise
    base_noise = generate_noise(img_shape, noise_config['type'], 1.0)

    # Modulate base noise by distance-based intensity map
    distance_noise = base_noise * final_noise_intensity_map
    distance_noise = distance_noise.astype(np.int16)

    # Apply noise only inside the buffered region
    noisy_image = image.copy().astype(np.int16)
    noise_applied_pixels = (noisy_region_mask == 1)

    if np.sum(noise_applied_pixels) > 0:
        for i in range(3):  # Apply to all RGB channels
            channel = noisy_image[:,:,i]
            channel[noise_applied_pixels] += distance_noise[noise_applied_pixels]
            np.clip(channel, 0, 255, out=channel)
            noisy_image[:,:,i] = channel

    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image, final_noise_intensity_map


def rotate_image_and_annotations(image, annotations, angle, interpolation='bilinear', fill_value=0, defer_clipping=False):
    """Rotate image and adjust annotations accordingly.

    Simple approach: Rotate the image, apply rotation matrix to annotation coordinates,
    clip to new rotated image bounds. No padding, no complex adjustments.

    Args:
        image: Input image
        annotations: List of annotation dictionaries
        angle: Rotation angle in degrees
        interpolation: Interpolation method ('bilinear', 'nearest', 'cubic')
        fill_value: Fill value for empty pixels
        defer_clipping: DEPRECATED - kept for backward compatibility, always clips to rotated bounds

    Returns:
        rotated_image: Rotated image (new dimensions based on rotation)
        rotated_annotations: Transformed and clipped annotations
        (new_width, new_height): Dimensions of rotated image
    """
    from .annotations import calculate_bbox_from_segmentation, calculate_area_from_segmentation

    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image dimensions after rotation
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))

    # Adjust rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Set interpolation method
    interp_method = cv2.INTER_LINEAR
    if interpolation == 'nearest':
        interp_method = cv2.INTER_NEAREST
    elif interpolation == 'cubic':
        interp_method = cv2.INTER_CUBIC

    # Rotate image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                  flags=interp_method,
                                  borderValue=fill_value)

    # Rotate annotations
    rotated_annotations = []
    input_count = len(annotations)
    skipped_count = 0
    skipped_reasons = {'invalid': 0, 'empty': 0, 'error': 0}

    for ann in annotations:
        segmentation = ann['segmentation'][0]
        if len(segmentation) < 6:
            # Skip degenerate polygons (less than 3 points)
            skipped_count += 1
            skipped_reasons['invalid'] += 1
            continue

        # Convert segmentation to points
        points = np.array([(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)])

        # Add homogeneous coordinate
        points_homogeneous = np.column_stack([points, np.ones(len(points))])

        # Apply rotation
        rotated_points = rotation_matrix @ points_homogeneous.T
        rotated_points = rotated_points.T[:, :2]  # Remove homogeneous coordinate

        # Create shapely polygon from rotated points
        try:
            rotated_polygon = ShapelyPolygon(rotated_points)

            # Fix invalid polygons with more robust repair
            if not rotated_polygon.is_valid:
                from tqdm import tqdm
                from shapely.validation import make_valid
                original_points = len(rotated_points)
                original_area = rotated_polygon.area
                tqdm.write(f"      DEBUG: Category {ann.get('category_id', '?')} invalid after rotation - {original_points} points, area={original_area:.1f}")

                # Try multiple repair strategies in sequence
                # Strategy 1: make_valid (best preservation of geometry)
                try:
                    fixed_polygon = make_valid(rotated_polygon)
                    if fixed_polygon.is_valid and not fixed_polygon.is_empty and fixed_polygon.area > original_area * 0.8:
                        rotated_polygon = fixed_polygon
                        geom_type = fixed_polygon.geom_type
                        if geom_type == 'Polygon':
                            tqdm.write(f"        → Repaired with make_valid: {len(list(fixed_polygon.exterior.coords))} points, area={fixed_polygon.area:.1f}")
                        else:
                            tqdm.write(f"        → Repaired with make_valid: {geom_type}, area={fixed_polygon.area:.1f}")
                except:
                    pass

                # Strategy 2: Buffer with zero distance (if make_valid didn't work or changed geometry too much)
                if not rotated_polygon.is_valid:
                    try:
                        fixed_polygon = rotated_polygon.buffer(0)
                        if fixed_polygon.is_valid and not fixed_polygon.is_empty and fixed_polygon.area > original_area * 0.5:
                            rotated_polygon = fixed_polygon
                            geom_type = fixed_polygon.geom_type
                            if geom_type == 'Polygon':
                                tqdm.write(f"        → Repaired with buffer(0): {len(list(fixed_polygon.exterior.coords))} points, area={fixed_polygon.area:.1f}")
                            else:
                                tqdm.write(f"        → Repaired with buffer(0): {geom_type}, area={fixed_polygon.area:.1f}")
                    except:
                        pass

                # Strategy 2: Very small positive buffer (0.01) to fix topology
                if not rotated_polygon.is_valid:
                    try:
                        fixed_polygon = rotated_polygon.buffer(0.01)
                        if fixed_polygon.is_valid and not fixed_polygon.is_empty:
                            rotated_polygon = fixed_polygon
                            tqdm.write(f"        → Repaired with buffer(0.01): {len(list(fixed_polygon.exterior.coords))} points, area={fixed_polygon.area:.1f}")
                    except:
                        pass

                # Strategy 3: Simplify with preserve_topology
                if not rotated_polygon.is_valid:
                    try:
                        fixed_polygon = rotated_polygon.simplify(0.5, preserve_topology=True)
                        if fixed_polygon.is_valid and not fixed_polygon.is_empty:
                            rotated_polygon = fixed_polygon
                            tqdm.write(f"        → Repaired with simplify(0.5): {len(list(fixed_polygon.exterior.coords))} points, area={fixed_polygon.area:.1f}")
                    except:
                        pass

                # Strategy 4: Simplify without preserve_topology (more aggressive)
                if not rotated_polygon.is_valid:
                    try:
                        fixed_polygon = rotated_polygon.simplify(1.0, preserve_topology=False)
                        if fixed_polygon.is_valid and not fixed_polygon.is_empty:
                            rotated_polygon = fixed_polygon
                            tqdm.write(f"        → Repaired with simplify(1.0): {len(list(fixed_polygon.exterior.coords))} points, area={fixed_polygon.area:.1f}")
                    except:
                        pass

                # Final check - only skip if still invalid
                if not rotated_polygon.is_valid:
                    skipped_count += 1
                    skipped_reasons['invalid'] += 1
                    from tqdm import tqdm
                    tqdm.write(f"      DEBUG: Category {ann.get('category_id', '?')} failed all repair attempts")
                    continue

            # Check if polygon is empty (but don't filter by area - rotation preserves area!)
            if rotated_polygon.is_empty:
                skipped_count += 1
                skipped_reasons['empty'] += 1
                continue

            # Clip polygon to rotated image boundaries
            image_bounds = box(0, 0, new_width, new_height)

            # Make intersection more robust
            try:
                clipped_polygon = rotated_polygon.intersection(image_bounds)
            except Exception as intersection_error:
                # Try buffering both polygons slightly
                try:
                    rotated_polygon = rotated_polygon.buffer(0.1)
                    clipped_polygon = rotated_polygon.intersection(image_bounds)
                except:
                    skipped_count += 1
                    skipped_reasons['error'] += 1
                    continue

            # Handle intersection results
            if clipped_polygon.is_empty:
                skipped_count += 1
                skipped_reasons['empty'] += 1
                continue

            # Handle MultiPolygon case - create separate annotation for each part
            # NO AREA FILTERING - rotation preserves area, so keep everything!
            polygons_to_process = []
            if clipped_polygon.geom_type == 'MultiPolygon':
                polygons_to_process = list(clipped_polygon.geoms)
            elif clipped_polygon.geom_type == 'Polygon':
                polygons_to_process = [clipped_polygon]
            elif clipped_polygon.geom_type == 'GeometryCollection':
                # Extract only Polygons (no area filter)
                polygons_to_process = [g for g in clipped_polygon.geoms if g.geom_type == 'Polygon']

            # Process each polygon part
            for poly in polygons_to_process:

                # Convert back to segmentation format
                exterior_coords = list(poly.exterior.coords[:-1])

                if len(exterior_coords) < 3:
                    continue

                # Flatten coordinates
                valid_points = []
                for x, y in exterior_coords:
                    x = max(0, min(x, new_width - 1))
                    y = max(0, min(y, new_height - 1))
                    valid_points.extend([x, y])

                if len(valid_points) < 6:
                    continue

                # Calculate new bounding box and area
                bbox = calculate_bbox_from_segmentation(valid_points)
                area = calculate_area_from_segmentation(valid_points)

                # Filter out degenerate polygons with zero area (collapsed to a line or point)
                if area <= 0:
                    continue

                # Create new annotation for this polygon part
                rotated_ann = ann.copy()
                rotated_ann['segmentation'] = [valid_points]
                rotated_ann['bbox'] = bbox
                rotated_ann['area'] = area

                rotated_annotations.append(rotated_ann)

        except Exception as e:
            # Log warning but continue processing other annotations
            from tqdm import tqdm
            tqdm.write(f"      WARNING: Failed to rotate annotation (category {ann.get('category_id', '?')}): {e}")
            skipped_count += 1
            skipped_reasons['error'] += 1
            continue

    # Log detailed summary if annotations were lost
    output_count = len(rotated_annotations)
    if output_count < input_count:
        from tqdm import tqdm
        lost_count = input_count - output_count
        reason_str = ', '.join([f"{k}: {v}" for k, v in skipped_reasons.items() if v > 0])
        tqdm.write(f"      Rotation: {input_count} → {output_count} annotations ({lost_count} lost - {reason_str})")

    return rotated_image, rotated_annotations, (new_width, new_height)


def clip_annotations_to_bounds(annotations, image_width, image_height):
    """Clip annotations to image boundaries and filter invalid ones.

    Args:
        annotations: List of annotation dictionaries with segmentation
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        List of clipped and valid annotations
    """
    from .annotations import calculate_bbox_from_segmentation, calculate_area_from_segmentation

    clipped_annotations = []
    input_count = len(annotations)
    skipped_count = 0
    skipped_reasons = {'invalid': 0, 'empty': 0, 'error': 0}

    for ann in annotations:
        segmentation = ann['segmentation'][0]
        if len(segmentation) < 6:
            # Skip degenerate polygons (less than 3 points)
            skipped_count += 1
            skipped_reasons['invalid'] += 1
            continue

        # Convert to points
        points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]

        # Create polygon
        try:
            polygon = ShapelyPolygon(points)

            # Fix invalid polygons with more robust repair
            if not polygon.is_valid:
                # Try multiple repair strategies in sequence
                # Strategy 1: Buffer with zero distance (most common fix)
                try:
                    fixed_polygon = polygon.buffer(0)
                    if fixed_polygon.is_valid and not fixed_polygon.is_empty:
                        polygon = fixed_polygon
                except:
                    pass

                # Strategy 2: Very small positive buffer (0.01) to fix topology
                if not polygon.is_valid:
                    try:
                        fixed_polygon = polygon.buffer(0.01)
                        if fixed_polygon.is_valid and not fixed_polygon.is_empty:
                            polygon = fixed_polygon
                    except:
                        pass

                # Strategy 3: Simplify with preserve_topology
                if not polygon.is_valid:
                    try:
                        fixed_polygon = polygon.simplify(0.5, preserve_topology=True)
                        if fixed_polygon.is_valid and not fixed_polygon.is_empty:
                            polygon = fixed_polygon
                    except:
                        pass

                # Strategy 4: Simplify without preserve_topology (more aggressive)
                if not polygon.is_valid:
                    try:
                        fixed_polygon = polygon.simplify(1.0, preserve_topology=False)
                        if fixed_polygon.is_valid and not fixed_polygon.is_empty:
                            polygon = fixed_polygon
                    except:
                        pass

                # Final check - only skip if still invalid
                if not polygon.is_valid:
                    skipped_count += 1
                    skipped_reasons['invalid'] += 1
                    from tqdm import tqdm
                    tqdm.write(f"      DEBUG: Clipping - Category {ann.get('category_id', '?')} failed all repair attempts")
                    continue

            if polygon.is_empty:
                skipped_count += 1
                skipped_reasons['empty'] += 1
                continue

            # NO AREA FILTER - keep all annotations
            # (Even for windows, let the tiny slivers through - user can filter later if needed)

            # Clip to image bounds
            image_bounds = box(0, 0, image_width, image_height)

            # Make intersection more robust
            try:
                clipped_polygon = polygon.intersection(image_bounds)
            except Exception as intersection_error:
                # Try buffering polygon slightly
                try:
                    polygon = polygon.buffer(0.1)
                    clipped_polygon = polygon.intersection(image_bounds)
                except:
                    skipped_count += 1
                    skipped_reasons['error'] += 1
                    continue

            if clipped_polygon.is_empty:
                skipped_count += 1
                skipped_reasons['empty'] += 1
                continue

            # Handle MultiPolygon case - NO AREA FILTERING
            polygons_to_process = []
            if clipped_polygon.geom_type == 'MultiPolygon':
                polygons_to_process = list(clipped_polygon.geoms)
            elif clipped_polygon.geom_type == 'Polygon':
                polygons_to_process = [clipped_polygon]
            elif clipped_polygon.geom_type == 'GeometryCollection':
                # Handle GeometryCollection by extracting only Polygons
                polygons_to_process = [g for g in clipped_polygon.geoms if g.geom_type == 'Polygon']

            # Process each polygon part
            for poly in polygons_to_process:

                # Convert back to segmentation format
                exterior_coords = list(poly.exterior.coords[:-1])

                if len(exterior_coords) < 3:
                    continue

                # Flatten and clamp coordinates
                valid_points = []
                for x, y in exterior_coords:
                    x = max(0, min(x, image_width - 1))
                    y = max(0, min(y, image_height - 1))
                    valid_points.extend([x, y])

                if len(valid_points) < 6:
                    continue

                # Calculate bbox and area
                bbox = calculate_bbox_from_segmentation(valid_points)
                area = calculate_area_from_segmentation(valid_points)

                # Filter out degenerate polygons with zero area (collapsed to a line or point)
                if area <= 0:
                    continue

                # Create clipped annotation
                clipped_ann = ann.copy()
                clipped_ann['segmentation'] = [valid_points]
                clipped_ann['bbox'] = bbox
                clipped_ann['area'] = area

                clipped_annotations.append(clipped_ann)

        except Exception as e:
            # Log warning but continue processing other annotations
            from tqdm import tqdm
            tqdm.write(f"      WARNING: Failed to clip annotation (category {ann.get('category_id', '?')}): {e}")
            skipped_count += 1
            skipped_reasons['error'] += 1
            continue

    # Log detailed summary if annotations were lost
    output_count = len(clipped_annotations)
    if output_count < input_count:
        from tqdm import tqdm
        lost_count = input_count - output_count
        reason_str = ', '.join([f"{k}: {v}" for k, v in skipped_reasons.items() if v > 0])
        tqdm.write(f"      Clipping: {input_count} → {output_count} annotations ({lost_count} lost - {reason_str})")

    return clipped_annotations


def scale_image_and_annotations(image, annotations, transform, max_size=1024, verbose=False):
    """Scale image and annotations to fit within max_size while maintaining aspect ratio."""
    from .annotations import calculate_bbox_from_segmentation, calculate_area_from_segmentation

    height, width = image.shape[:2]

    # Calculate scale factor
    scale_factor = 1.0
    if width > max_size or height > max_size:
        scale_factor = min(max_size / width, max_size / height)

    # If no scaling needed, return original
    if scale_factor >= 1.0:
        return image, annotations, transform, scale_factor

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    if verbose:
        print(f"    Scaling from {width}x{height} to {new_width}x{new_height} (factor: {scale_factor:.3f})")

    # Resize image
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Scale annotations
    scaled_annotations = []
    for ann in annotations:
        segmentation = ann['segmentation'][0]
        if len(segmentation) < 6:
            continue

        # Scale segmentation coordinates
        scaled_segmentation = []
        for i in range(0, len(segmentation), 2):
            x = segmentation[i] * scale_factor
            y = segmentation[i+1] * scale_factor
            scaled_segmentation.extend([x, y])

        # Recalculate bbox and area
        bbox = calculate_bbox_from_segmentation(scaled_segmentation)
        area = calculate_area_from_segmentation(scaled_segmentation)

        # Skip if too small after scaling
        if area < 10:
            continue

        scaled_ann = ann.copy()
        scaled_ann['segmentation'] = [scaled_segmentation]
        scaled_ann['bbox'] = bbox
        scaled_ann['area'] = area

        scaled_annotations.append(scaled_ann)

    # Scale the transform
    scaled_transform = rasterio.Affine(
        transform.a / scale_factor,  # Scale pixel size
        transform.b,
        transform.c,  # Keep origin
        transform.d,
        transform.e / scale_factor,  # Scale pixel size
        transform.f   # Keep origin
    )

    return scaled_image, scaled_annotations, scaled_transform, scale_factor


def generate_random_angles(count, angle_range):
    """Generate random rotation angles within the specified range."""
    min_angle, max_angle = angle_range
    return [random.uniform(min_angle, max_angle) for _ in range(count)]
