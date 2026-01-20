"""
Person detector using YOLOX/RTMDet ONNX models.

Requires ONNX Runtime for inference.
No fallback - models must be downloaded.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "Please install onnxruntime or onnxruntime-gpu:\n"
        "  pip install onnxruntime-gpu  # For GPU\n"
        "  pip install onnxruntime       # For CPU"
    )

from mocap_app.types import BoundingBox

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    YOLOX/RTMDet-based person detector using ONNX Runtime.

    Requires downloaded ONNX model - no fallback implementation.
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cpu",
    ):
        """
        Initialize person detector.

        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Model input size (height, width)
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device

        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please run: python download_models.py"
            )

        logger.info(f"Loading detector model: {model_path.name}")

        # Initialize ONNX Runtime session
        providers = self._get_providers(device)
        logger.info(f"Using providers: {providers}")

        self.session = ort.InferenceSession(str(model_path), providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"✓ Detector loaded successfully")
        logger.info(f"  Input: {self.input_name}")
        logger.info(f"  Outputs: {self.output_names}")

    def _get_providers(self, device: str) -> List[str]:
        """Get ONNX Runtime execution providers based on device."""
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "directml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif device == "coreml":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    def preprocess(
        self, image: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.float32], float, float, Tuple[int, int]]:
        """
        Preprocess image for detection.

        Args:
            image: Input image (BGR format)

        Returns:
            Preprocessed tensor, scale_x, scale_y, padding
        """
        orig_height, orig_width = image.shape[:2]
        target_height, target_width = self.input_size

        # Resize with aspect ratio preservation
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        padded[:new_height, :new_width] = resized

        # Convert to RGB and normalize
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32)

        # Normalize (ImageNet mean/std)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        padded = (padded - mean) / std

        # Transpose to CHW format and add batch dimension
        tensor = padded.transpose(2, 0, 1)[None, ...].astype(np.float32)

        scale_x = orig_width / new_width
        scale_y = orig_height / new_height

        return tensor, scale_x, scale_y, (orig_height, orig_width)

    def postprocess(
        self,
        outputs: List[NDArray],
        scale_x: float,
        scale_y: float,
        orig_shape: Tuple[int, int],
    ) -> List[BoundingBox]:
        """
        Post-process detection outputs.

        Args:
            outputs: Model outputs
            scale_x: X-axis scale factor
            scale_y: Y-axis scale factor
            orig_shape: Original image shape (height, width)

        Returns:
            List of detected person bounding boxes
        """
        # YOLOX/RTMDet output format: [batch, num_boxes, 85]
        # [x1, y1, x2, y2, obj_conf, class_0_conf, class_1_conf, ..., class_79_conf]
        detections = outputs[0][0]  # Remove batch dimension

        persons = []

        for det in detections:
            x1, y1, x2, y2 = det[:4]
            obj_conf = det[4]

            # Class confidences (COCO format: class 0 is 'person')
            class_confs = det[5:]
            class_id = np.argmax(class_confs)
            class_conf = class_confs[class_id]

            # Combined confidence
            confidence = obj_conf * class_conf

            # Filter by person class and confidence
            if class_id == 0 and confidence >= self.confidence_threshold:
                # Scale coordinates back to original image size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # Clip to image boundaries
                orig_height, orig_width = orig_shape
                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(0, min(x2, orig_width))
                y2 = max(0, min(y2, orig_height))

                if x2 > x1 and y2 > y1:  # Valid box
                    persons.append(
                        BoundingBox(x1, y1, x2, y2, float(confidence))
                    )

        # Apply NMS
        if len(persons) > 1:
            persons = self._nms(persons, self.nms_threshold)

        logger.debug(f"Detected {len(persons)} persons")

        return persons

    def _nms(self, boxes: List[BoundingBox], threshold: float) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []

        # Convert to numpy array
        boxes_array = np.array([box.to_xyxy() for box in boxes])
        scores = np.array([box.confidence for box in boxes])

        # Compute areas
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by confidence
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def detect(self, image: NDArray[np.uint8]) -> List[BoundingBox]:
        """
        Detect persons in an image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detected person bounding boxes
        """
        # Preprocess
        tensor, scale_x, scale_y, orig_shape = self.preprocess(image)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: tensor})

        # Postprocess
        persons = self.postprocess(outputs, scale_x, scale_y, orig_shape)

        return persons

    def __call__(self, image: NDArray[np.uint8]) -> List[BoundingBox]:
        """Alias for detect()."""
        return self.detect(image)
