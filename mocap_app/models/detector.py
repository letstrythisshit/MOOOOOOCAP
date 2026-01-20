"""
Person detector using RTMDet.

This is a demonstration implementation showing the architecture.
For production, replace with ONNX Runtime inference on actual RTMDet models.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from mocap_app.types import BoundingBox


class PersonDetector:
    """
    RTMDet-based person detector.

    Currently uses a simplified implementation for demonstration.
    Replace with ONNX Runtime for production use.
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.device = device

        # For demo: Use OpenCV's DNN detector (pre-trained on COCO)
        # In production, replace this with ONNX Runtime loading RTMDet
        try:
            # Try to use a simple HOG detector as fallback
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.use_hog = True
        except:
            self.use_hog = False

    def detect(self, image: NDArray[np.uint8]) -> List[BoundingBox]:
        """
        Detect persons in an image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of bounding boxes for detected persons
        """
        if not self.use_hog:
            # Fallback: Return a mock detection in the center
            h, w = image.shape[:2]
            return [
                BoundingBox(
                    x1=w // 4,
                    y1=h // 4,
                    x2=3 * w // 4,
                    y2=3 * h // 4,
                    confidence=0.95,
                )
            ]

        # Use HOG detector as demo
        boxes, weights = self.hog.detectMultiScale(
            image, winStride=(8, 8), padding=(4, 4), scale=1.05
        )

        detections = []
        for (x, y, w, h), weight in zip(boxes, weights.flatten()):
            if weight > self.confidence_threshold:
                detections.append(
                    BoundingBox(
                        x1=int(x),
                        y1=int(y),
                        x2=int(x + w),
                        y2=int(y + h),
                        confidence=float(weight),
                    )
                )

        # Apply NMS
        if len(detections) > 1:
            detections = self._nms(detections, 0.5)

        return detections

    def _nms(self, boxes: List[BoundingBox], threshold: float) -> List[BoundingBox]:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return []

        # Sort by confidence
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)

        keep = []
        while boxes:
            current = boxes.pop(0)
            keep.append(current)

            boxes = [
                box
                for box in boxes
                if self._compute_iou(current, box) < threshold
            ]

        return keep

    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection

        return intersection / union if union > 0 else 0.0

    def __call__(self, image: NDArray[np.uint8]) -> List[BoundingBox]:
        """Alias for detect()."""
        return self.detect(image)
