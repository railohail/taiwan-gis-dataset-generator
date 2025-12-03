"""
Annotation data models.

Data classes for representing annotations in various formats.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class BoundingBox:
    """
    Bounding box in COCO format.

    Format: [x, y, width, height]
    - x, y: top-left corner
    - width, height: box dimensions
    """
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_segmentation(cls, segmentation: list[float]) -> 'BoundingBox':
        """
        Calculate bounding box from segmentation coordinates.

        Args:
            segmentation: Flat list of [x1, y1, x2, y2, ..., xn, yn]

        Returns:
            BoundingBox instance
        """
        if not segmentation or len(segmentation) < 2:
            return cls(0, 0, 0, 0)

        x_coords = segmentation[0::2]
        y_coords = segmentation[1::2]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return cls(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min
        )

    def to_list(self) -> list[float]:
        """Convert to COCO format list."""
        return [self.x, self.y, self.width, self.height]

    def to_yolo_normalized(self, image_width: int, image_height: int) -> list[float]:
        """
        Convert to YOLO normalized format.

        YOLO format: [x_center, y_center, width, height] (all 0-1)

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            [x_center, y_center, width, height] normalized to 0-1
        """
        x_center = (self.x + self.width / 2) / image_width
        y_center = (self.y + self.height / 2) / image_height
        width_norm = self.width / image_width
        height_norm = self.height / image_height

        # Clamp to [0, 1]
        return [
            max(0.0, min(1.0, x_center)),
            max(0.0, min(1.0, y_center)),
            max(0.0, min(1.0, width_norm)),
            max(0.0, min(1.0, height_norm))
        ]

    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height


@dataclass
class Annotation:
    """
    Annotation for a single object instance.

    Compatible with both COCO and YOLO formats.
    """
    annotation_id: int
    image_id: int
    category_id: int
    segmentation: list[float] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    area: float = 0.0
    iscrowd: int = 0

    def __post_init__(self):
        """Calculate bbox and area if not provided."""
        if self.bbox is None and self.segmentation:
            self.bbox = BoundingBox.from_segmentation(self.segmentation)

        if self.area == 0.0 and self.segmentation:
            self.area = self._calculate_polygon_area()

    def _calculate_polygon_area(self) -> float:
        """
        Calculate polygon area using the shoelace formula.

        Returns:
            Area in square pixels
        """
        if len(self.segmentation) < 6:
            return 0.0

        points = [
            (self.segmentation[i], self.segmentation[i + 1])
            for i in range(0, len(self.segmentation), 2)
        ]

        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2.0

    def to_coco_dict(self, include_segmentation: bool = True) -> dict[str, Any]:
        """
        Convert to COCO format dictionary.

        Args:
            include_segmentation: Whether to include segmentation field

        Returns:
            COCO annotation dictionary
        """
        result = {
            "id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox.to_list() if self.bbox else [0, 0, 0, 0],
            "area": self.area,
            "iscrowd": self.iscrowd
        }

        if include_segmentation and self.segmentation:
            result["segmentation"] = [self.segmentation]

        return result

    def to_yolo_bbox(self, image_width: int, image_height: int) -> str:
        """
        Convert to YOLO bounding box format.

        Format: "class_id x_center y_center width height"

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            YOLO format string
        """
        if self.bbox is None:
            return ""

        bbox_norm = self.bbox.to_yolo_normalized(image_width, image_height)
        return f"{self.category_id} {bbox_norm[0]:.6f} {bbox_norm[1]:.6f} {bbox_norm[2]:.6f} {bbox_norm[3]:.6f}"

    def to_yolo_segmentation(self, image_width: int, image_height: int) -> str:
        """
        Convert to YOLO segmentation format.

        Format: "class_id x1 y1 x2 y2 ... xn yn" (all normalized)

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            YOLO format string
        """
        if not self.segmentation or len(self.segmentation) < 6:
            return ""

        # Normalize coordinates
        normalized = []
        for i in range(0, len(self.segmentation), 2):
            x = max(0.0, min(1.0, self.segmentation[i] / image_width))
            y = max(0.0, min(1.0, self.segmentation[i + 1] / image_height))
            normalized.extend([x, y])

        coords_str = ' '.join([f"{coord:.6f}" for coord in normalized])
        return f"{self.category_id} {coords_str}"

    @classmethod
    def from_shapely_polygon(
        cls,
        polygon,
        annotation_id: int,
        image_id: int,
        category_id: int,
        transform,
        image_shape: tuple[int, int]
    ) -> Optional['Annotation']:
        """
        Create annotation from Shapely polygon.

        Args:
            polygon: Shapely Polygon object
            annotation_id: Annotation ID
            image_id: Image ID
            category_id: Category/class ID
            transform: Rasterio transform for coordinate conversion
            image_shape: (height, width) of image

        Returns:
            Annotation instance or None if invalid
        """
        # Convert geometry to pixel coordinates
        pixel_coords = []
        for x, y in polygon.exterior.coords:
            px, py = ~transform * (x, y)
            pixel_coords.append([px, py])

        if len(pixel_coords) < 3:
            return None

        # Convert to segmentation format (excluding last duplicate)
        segmentation = []
        for px, py in pixel_coords[:-1]:
            # Clamp to image bounds
            px_clamped = max(0, min(px, image_shape[1] - 1))
            py_clamped = max(0, min(py, image_shape[0] - 1))
            segmentation.extend([px_clamped, py_clamped])

        if len(segmentation) < 6:
            return None

        return cls(
            annotation_id=annotation_id,
            image_id=image_id,
            category_id=category_id,
            segmentation=segmentation
        )
