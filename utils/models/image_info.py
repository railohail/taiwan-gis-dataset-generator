"""
Image metadata models.

Data classes for representing image information.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageInfo:
    """
    Image metadata for dataset generation.

    Compatible with COCO format image entries.
    """
    image_id: int
    file_name: str
    width: int
    height: int
    date_captured: str = ""
    license: int = 1
    coco_url: str = ""
    flickr_url: str = ""

    def to_coco_dict(self) -> dict[str, Any]:
        """
        Convert to COCO format dictionary.

        Returns:
            COCO image dictionary
        """
        return {
            "id": self.image_id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "date_captured": self.date_captured,
            "license": self.license,
            "coco_url": self.coco_url,
            "flickr_url": self.flickr_url
        }

    @classmethod
    def from_array(
        cls,
        image_id: int,
        file_name: str,
        image_array
    ) -> 'ImageInfo':
        """
        Create ImageInfo from numpy array.

        Args:
            image_id: Image ID
            file_name: Image filename
            image_array: Numpy array with shape (H, W, C) or (H, W)

        Returns:
            ImageInfo instance
        """
        if len(image_array.shape) == 2:
            height, width = image_array.shape
        else:
            height, width = image_array.shape[:2]

        return cls(
            image_id=image_id,
            file_name=file_name,
            width=width,
            height=height
        )

    @classmethod
    def from_file(
        cls,
        image_id: int,
        file_path: Path
    ) -> 'ImageInfo':
        """
        Create ImageInfo from image file.

        Args:
            image_id: Image ID
            file_path: Path to image file

        Returns:
            ImageInfo instance

        Raises:
            ImportError: If opencv-python is not installed
            FileNotFoundError: If image file doesn't exist
        """
        import cv2

        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Read image to get dimensions
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"Failed to read image: {file_path}")

        height, width = img.shape[:2]

        return cls(
            image_id=image_id,
            file_name=file_path.name,
            width=width,
            height=height
        )

    def validate(self) -> bool:
        """
        Validate image metadata.

        Returns:
            True if valid, False otherwise
        """
        return (
            self.image_id > 0 and
            self.width > 0 and
            self.height > 0 and
            bool(self.file_name)
        )
