"""
RTMPose-based whole-body pose estimation using ONNX Runtime.

RTMPose is a state-of-the-art real-time pose estimator supporting:
- Body pose (17 keypoints - COCO format)
- Whole-body pose (133 keypoints - body + face + hands)
- Hand pose (21 keypoints per hand)

License: Apache 2.0 from OpenMMLab
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Please install onnxruntime or onnxruntime-gpu")

from mocap_app.core.types import BBox, FingerArticulation, HandPose, WholeBodyPose


class WholeBodyPoseEstimator:
    """
    Real-time whole-body pose estimation using RTMPose.

    Estimates 133 keypoints:
    - 17 body keypoints (COCO format)
    - 6 foot keypoints
    - 68 face landmarks
    - 21 left hand keypoints
    - 21 right hand keypoints
    """

    # COCO body keypoint indices
    BODY_KEYPOINTS = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    # Hand keypoint indices (within hand region)
    HAND_KEYPOINTS = {
        "wrist": 0,
        "thumb_1": 1, "thumb_2": 2, "thumb_3": 3, "thumb_4": 4,
        "index_1": 5, "index_2": 6, "index_3": 7, "index_4": 8,
        "middle_1": 9, "middle_2": 10, "middle_3": 11, "middle_4": 12,
        "ring_1": 13, "ring_2": 14, "ring_3": 15, "ring_4": 16,
        "pinky_1": 17, "pinky_2": 18, "pinky_3": 19, "pinky_4": 20,
    }

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.3,
        input_size: Tuple[int, int] = (288, 384),
        device: str = "cpu",
    ):
        """
        Initialize whole-body pose estimator.

        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for keypoints
            input_size: Model input size (height, width)
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

        # Initialize ONNX Runtime session
        providers = self._get_providers(device)
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _get_providers(self, device: str) -> List[str]:
        """Get ONNX Runtime execution providers."""
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "directml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif device == "coreml":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    def preprocess(
        self,
        image: NDArray[np.uint8],
        bbox: BBox,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Preprocess image crop for pose estimation.

        Args:
            image: Full image (BGR format)
            bbox: Person bounding box

        Returns:
            Preprocessed tensor and transformation matrix
        """
        # Expand bbox slightly for better context
        center = np.array(bbox.center, dtype=np.float32)
        scale = max(bbox.width, bbox.height) * 1.25

        # Compute transformation matrix
        trans_matrix = self._get_transform_matrix(
            center, scale, self.input_size
        )

        # Warp image
        target_height, target_width = self.input_size
        warped = cv2.warpAffine(
            image,
            trans_matrix,
            (target_width, target_height),
            flags=cv2.INTER_LINEAR,
        )

        # Convert to RGB and normalize
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        warped = warped.astype(np.float32) / 255.0

        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        warped = (warped - mean) / std

        # Transpose to CHW and add batch dimension
        tensor = warped.transpose(2, 0, 1)[None, ...].astype(np.float32)

        return tensor, trans_matrix

    def _get_transform_matrix(
        self,
        center: NDArray[np.float32],
        scale: float,
        output_size: Tuple[int, int],
    ) -> NDArray[np.float32]:
        """Compute affine transformation matrix."""
        target_height, target_width = output_size

        # Scale and translation
        scale_x = target_width / scale
        scale_y = target_height / scale

        # Transformation matrix
        matrix = np.array([
            [scale_x, 0, -center[0] * scale_x + target_width / 2],
            [0, scale_y, -center[1] * scale_y + target_height / 2],
        ], dtype=np.float32)

        return matrix

    def _transform_keypoints(
        self,
        keypoints: NDArray[np.float32],
        trans_matrix: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Transform keypoints from normalized space to original image space."""
        # Invert transformation matrix
        trans_matrix_full = np.vstack([trans_matrix, [0, 0, 1]])
        inv_matrix = np.linalg.inv(trans_matrix_full)[:2]

        # Transform keypoints
        num_keypoints = keypoints.shape[0]
        coords = keypoints[:, :2]  # (N, 2)
        confidences = keypoints[:, 2:3]  # (N, 1)

        # Add homogeneous coordinate
        ones = np.ones((num_keypoints, 1), dtype=np.float32)
        coords_homo = np.hstack([coords, ones])  # (N, 3)

        # Apply inverse transformation
        transformed = (inv_matrix @ coords_homo.T).T  # (N, 2)

        # Combine with confidences
        result = np.hstack([transformed, confidences])

        return result

    def postprocess(
        self,
        outputs: List[NDArray],
        trans_matrix: NDArray[np.float32],
    ) -> Optional[WholeBodyPose]:
        """
        Post-process model outputs to extract keypoints.

        Args:
            outputs: Model outputs (heatmaps)
            trans_matrix: Transformation matrix used during preprocessing

        Returns:
            WholeBodyPose or None if no valid pose detected
        """
        # RTMPose outputs SimCC format: [x_coords, y_coords]
        # Shape: (batch, num_keypoints, coord_dim)
        if len(outputs) == 2:
            x_coords = outputs[0][0]  # (133, W)
            y_coords = outputs[1][0]  # (133, H)

            num_keypoints = x_coords.shape[0]
            keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)

            target_height, target_width = self.input_size

            for i in range(num_keypoints):
                # Get maximum response
                x_idx = np.argmax(x_coords[i])
                y_idx = np.argmax(y_coords[i])

                x_conf = x_coords[i, x_idx]
                y_conf = y_coords[i, y_idx]

                # Convert to coordinates
                x = (x_idx / x_coords.shape[1]) * target_width
                y = (y_idx / y_coords.shape[1]) * target_height

                # Confidence is geometric mean of x and y confidences
                conf = np.sqrt(x_conf * y_conf)

                keypoints[i] = [x, y, conf]

        else:
            # Fallback: heatmap format
            heatmaps = outputs[0][0]  # (133, H, W)
            num_keypoints = heatmaps.shape[0]
            target_height, target_width = self.input_size

            keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)

            for i in range(num_keypoints):
                heatmap = heatmaps[i]
                y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)

                # Sub-pixel refinement (optional)
                conf = heatmap[y_idx, x_idx]

                x = (x_idx / heatmap.shape[1]) * target_width
                y = (y_idx / heatmap.shape[0]) * target_height

                keypoints[i] = [x, y, conf]

        # Transform keypoints back to original image space
        keypoints = self._transform_keypoints(keypoints, trans_matrix)

        # Compute overall score
        valid_keypoints = keypoints[keypoints[:, 2] > self.confidence_threshold]
        if len(valid_keypoints) == 0:
            return None

        score = float(np.mean(valid_keypoints[:, 2]))

        # Create WholeBodyPose
        pose = WholeBodyPose(keypoints=keypoints, score=score)

        # Extract and analyze hands
        pose.left_hand = self._extract_hand(pose.left_hand_keypoints, "left")
        pose.right_hand = self._extract_hand(pose.right_hand_keypoints, "right")

        return pose

    def _extract_hand(self, hand_keypoints: NDArray[np.float32], side: str) -> Optional[HandPose]:
        """Extract hand pose and analyze finger articulation."""
        # Check if hand is visible
        valid_kpts = hand_keypoints[hand_keypoints[:, 2] > self.confidence_threshold]
        if len(valid_kpts) < 10:  # Need at least half the keypoints
            return None

        # Analyze finger articulation
        articulation = self._analyze_finger_articulation(hand_keypoints)

        return HandPose(
            keypoints=hand_keypoints,
            articulation=articulation,
            side=side,
        )

    def _analyze_finger_articulation(self, hand_kpts: NDArray[np.float32]) -> FingerArticulation:
        """
        Analyze detailed finger articulation from hand keypoints.

        Returns per-finger curl values and overall hand state.
        """
        # Hand keypoint indices
        wrist = hand_kpts[0, :2]

        # Finger tip indices
        thumb_tip = hand_kpts[4, :2]
        index_tip = hand_kpts[8, :2]
        middle_tip = hand_kpts[12, :2]
        ring_tip = hand_kpts[16, :2]
        pinky_tip = hand_kpts[20, :2]

        # Finger base indices (MCP joints)
        thumb_base = hand_kpts[2, :2]
        index_base = hand_kpts[5, :2]
        middle_base = hand_kpts[9, :2]
        ring_base = hand_kpts[13, :2]
        pinky_base = hand_kpts[17, :2]

        # Compute finger curl (0 = straight, 1 = fully curled)
        def compute_curl(tip, base, wrist):
            # Distance from tip to wrist vs base to wrist
            tip_dist = np.linalg.norm(tip - wrist)
            base_dist = np.linalg.norm(base - wrist)

            if base_dist < 1e-6:
                return 0.5

            # Curl ratio
            ratio = tip_dist / base_dist
            # Normalize to [0, 1] where 1 is fully curled
            curl = max(0, min(1, 1.5 - ratio))

            return float(curl)

        thumb_curl = compute_curl(thumb_tip, thumb_base, wrist)
        index_curl = compute_curl(index_tip, index_base, wrist)
        middle_curl = compute_curl(middle_tip, middle_base, wrist)
        ring_curl = compute_curl(ring_tip, ring_base, wrist)
        pinky_curl = compute_curl(pinky_tip, pinky_base, wrist)

        # Compute finger spread
        finger_tips = np.array([index_tip, middle_tip, ring_tip, pinky_tip])
        pairwise_dists = []
        for i in range(len(finger_tips) - 1):
            dist = np.linalg.norm(finger_tips[i] - finger_tips[i + 1])
            pairwise_dists.append(dist)

        avg_spread = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
        # Normalize spread (typical range: 20-60 pixels)
        spread = max(0, min(1, (avg_spread - 20) / 40))

        # Determine overall hand state
        avg_curl = (index_curl + middle_curl + ring_curl + pinky_curl) / 4

        if avg_curl > 0.7:
            state = "fist"
        elif avg_curl < 0.3 and spread > 0.5:
            state = "open"
        elif index_curl < 0.3 and middle_curl > 0.6:
            state = "point"
        elif index_curl < 0.3 and middle_curl < 0.3 and ring_curl > 0.6:
            state = "peace"
        elif thumb_curl < 0.3 and index_curl > 0.6:
            state = "thumbs_up"
        elif avg_curl < 0.5:
            state = "partial"
        else:
            state = "closed"

        return FingerArticulation(
            thumb_curl=thumb_curl,
            index_curl=index_curl,
            middle_curl=middle_curl,
            ring_curl=ring_curl,
            pinky_curl=pinky_curl,
            spread=spread,
            state=state,
        )

    def estimate(self, image: NDArray[np.uint8], bbox: BBox) -> Optional[WholeBodyPose]:
        """
        Estimate whole-body pose for a person.

        Args:
            image: Full image (BGR format)
            bbox: Person bounding box

        Returns:
            WholeBodyPose or None if estimation failed
        """
        # Preprocess
        tensor, trans_matrix = self.preprocess(image, bbox)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: tensor})

        # Postprocess
        pose = self.postprocess(outputs, trans_matrix)

        return pose

    def __call__(self, image: NDArray[np.uint8], bbox: BBox) -> Optional[WholeBodyPose]:
        """Alias for estimate()."""
        return self.estimate(image, bbox)
