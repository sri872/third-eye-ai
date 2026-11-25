"""OpenCV Zoo LPD-YuNet license-plate detector wrapper.

Model: ``license_plate_detection_lpd_yunet_2023mar.onnx`` from OpenCV Zoo
(Apache-2.0, contributed by Dong Xu / WATRIX). The exact SHA-256 of the vendored
file is verified before loading. Inference uses OpenCV ``dnn`` on CPU, so it
works on macOS/Apple Silicon without extra runtime packages.

Postprocessing follows the upstream ``lpd_yunet.py`` reference implementation:
prior boxes, score fusion ``sqrt(class_score * iou_score)``, decoding with the
documented ``[0.1, 0.2]`` variances, and OpenCV DNN NMS.

Limitation: the upstream model card states the model was trained on Chinese
plates. It is used here as candidate localisation only; OCR evidence, not this
detector score, decides whether registration text is readable.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
import hashlib
from typing import Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "plate_detector"
    / "license_plate_detection_lpd_yunet_2023mar.onnx"
)
MODEL_SHA256 = "6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7"
METHOD = "YUNET_LPD_2023MAR"
DETECTION_THRESHOLD = 0.25
NMS_THRESHOLD = 0.30
MODEL_INPUT_WIDTH = 320
MODEL_INPUT_HEIGHT = 240
_VARIANCE = [0.1, 0.2]
_MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
_STEPS = [8, 16, 32, 64]
_OUTPUT_NAMES = ["loc", "conf", "iou"]


def verify_model_file(path: Path = MODEL_PATH) -> bool:
    """Return True only when the vendored ONNX matches the official hash."""
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == MODEL_SHA256


def _build_priors(width: int, height: int) -> np.ndarray:
    """Reproduce the upstream YuNet prior-box generation for a given input size."""
    f2 = [int(int((height + 1) / 2) / 2), int(int((width + 1) / 2) / 2)]
    f3 = [int(f2[0] / 2), int(f2[1] / 2)]
    f4 = [int(f3[0] / 2), int(f3[1] / 2)]
    f5 = [int(f4[0] / 2), int(f4[1] / 2)]
    f6 = [int(f5[0] / 2), int(f5[1] / 2)]
    priors: list[list[float]] = []
    for index, feature_map in enumerate([f3, f4, f5, f6]):
        for row, col in product(range(feature_map[0]), range(feature_map[1])):
            for min_size in _MIN_SIZES[index]:
                priors.append(
                    [
                        (col + 0.5) * _STEPS[index] / width,
                        (row + 0.5) * _STEPS[index] / height,
                        min_size / width,
                        min_size / height,
                    ]
                )
    return np.asarray(priors, dtype=np.float32)


class YuNetPlateDetector:
    """Small wrapper around the vendored OpenCV Zoo ONNX detector."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        conf_threshold: float = DETECTION_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> None:
        if not verify_model_file(model_path):
            raise ValueError("LPD-YuNet ONNX missing or hash mismatch")
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.net = cv2.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._priors: dict[tuple[int, int], np.ndarray] = {}

    def _priors_for(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        if key not in self._priors:
            self._priors[key] = _build_priors(width, height)
        return self._priors[key]

    def _decode(self, outputs: Sequence[np.ndarray], width: int, height: int) -> np.ndarray:
        loc, conf, iou = outputs
        priors = self._priors_for(width, height)
        if loc.shape[0] != priors.shape[0]:
            raise ValueError(
                f"YuNet output/prior mismatch: {loc.shape[0]} outputs, "
                f"{priors.shape[0]} priors at {width}x{height}"
            )
        cls_scores = conf[:, 1]
        iou_scores = np.clip(iou[:, 0], 0.0, 1.0)
        scores = np.sqrt(cls_scores * iou_scores)
        centre = priors[:, 0:2]
        size = priors[:, 2:4]
        scale = np.asarray([width, height], dtype=np.float32)
        points = np.hstack(
            (
                (centre + loc[:, 4:6] * _VARIANCE[0] * size) * scale,
                (centre + loc[:, 6:8] * _VARIANCE[0] * size) * scale,
                (centre + loc[:, 10:12] * _VARIANCE[0] * size) * scale,
                (centre + loc[:, 12:14] * _VARIANCE[0] * size) * scale,
            )
        )
        return np.hstack((points, scores[:, np.newaxis])).astype(np.float32)

    def detect(self, crop: np.ndarray) -> tuple[tuple[tuple[int, int, int, int], float], ...]:
        """Return axis-aligned candidate boxes in crop-local pixels and real scores."""
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return ()
        source_h, source_w = crop.shape[:2]
        # Fixed upstream input size keeps prior/output alignment deterministic.
        width, height = MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT
        resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(resized)
        self.net.setInput(blob)
        outputs = self.net.forward(_OUTPUT_NAMES)
        detections = self._decode(outputs, width, height)

        keep = cv2.dnn.NMSBoxes(
            bboxes=detections[:, 0:4].tolist(),
            scores=detections[:, -1].tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
        )
        if len(keep) == 0:
            return ()

        boxes: list[tuple[tuple[int, int, int, int], float]] = []
        x_scale = source_w / width
        y_scale = source_h / height
        for index in np.asarray(keep).reshape(-1):
            detection = detections[int(index)]
            points = detection[:8].reshape(4, 2)
            points[:, 0] *= x_scale
            points[:, 1] *= y_scale
            x, y, box_w, box_h = cv2.boundingRect(np.round(points).astype(np.int32))
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(source_w, x + box_w)
            y1 = min(source_h, y + box_h)
            if x1 <= x0 or y1 <= y0:
                continue
            if (x1 - x0) < 8 or (y1 - y0) < 4:
                continue
            boxes.append(((x0, y0, x1, y1), float(detection[-1])))
        return tuple(boxes)


@lru_cache(maxsize=1)
def get_yunet_detector() -> Optional[YuNetPlateDetector]:
    """Load the verified detector once, returning None if it cannot be loaded."""
    try:
        return YuNetPlateDetector()
    except Exception:
        return None
