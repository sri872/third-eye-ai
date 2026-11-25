"""Computer Vision Detector module for Project Third-Eye.

Loads Faster R-CNN ResNet-50 FPN (COCO pretrained) with lazy loading.
Supports Apple Silicon MPS acceleration with automatic CPU fallback.
Filters for traffic and accident-relevant COCO classes and returns Detection schemas.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
import uuid

import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from core.schemas import Detection

logger = logging.getLogger("third_eye.cv.detector")

# Allowed COCO classes for accident and triage detection
ALLOWED_CLASSES: Set[str] = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}

# Global singleton storage for lazy loading
_MODEL: Optional[torch.nn.Module] = None
_MODEL_DEVICE: Optional[torch.device] = None
_CATEGORIES: Optional[List[str]] = None


def get_target_device() -> torch.device:
    """Select the best available compute device: MPS if supported, else CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            # Verification probe on MPS
            probe = torch.zeros(1, device="mps")
            del probe
            return torch.device("mps")
        except Exception as e:
            logger.warning("MPS probe failed (%s); falling back to CPU.", e)
            return torch.device("cpu")
    return torch.device("cpu")


def get_detector_model() -> Tuple[torch.nn.Module, torch.device, List[str]]:
    """Lazy-load Faster R-CNN ResNet-50 FPN model, weights, and categories."""
    global _MODEL, _MODEL_DEVICE, _CATEGORIES

    if _MODEL is None:
        device = get_target_device()
        print(f"[Third-Eye CV] Initializing Faster R-CNN ResNet-50 FPN on device: {device.type.upper()}")
        
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _CATEGORIES = weights.meta["categories"]
        
        try:
            model = fasterrcnn_resnet50_fpn(weights=weights)
            model.to(device)
            model.eval()
            _MODEL = model
            _MODEL_DEVICE = device
        except Exception as e:
            if device.type == "mps":
                print(f"[Third-Eye CV] MPS initialization failed ({e}). Falling back to CPU.")
                device = torch.device("cpu")
                model = fasterrcnn_resnet50_fpn(weights=weights)
                model.to(device)
                model.eval()
                _MODEL = model
                _MODEL_DEVICE = device
            else:
                raise e

    return _MODEL, _MODEL_DEVICE, _CATEGORIES


def preprocess_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    """Convert path, numpy array, or PIL image into a normalized PyTorch tensor [C, H, W] in [0, 1].

    Returns:
        tensor: float tensor with values in [0, 1], shape [3, H, W]
        (width, height): image dimensions
        rgb_array: uint8 numpy array in RGB format
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        # Respect a real EXIF orientation tag before inference. The demo photos
        # have no such tag, so this is a no-op for them.
        pil_img = ImageOps.exif_transpose(Image.open(str(img_path))).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = ImageOps.exif_transpose(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Handle numpy image: assume BGR if from cv2, or RGB if 3 channels
        if image_input.ndim != 3 or image_input.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image array, got shape {image_input.shape}")
        # If array is uint8, convert BGR to RGB
        pil_img = Image.fromarray(image_input[:, :, ::-1]) if image_input.dtype == np.uint8 else Image.fromarray((image_input * 255).astype(np.uint8)[:, :, ::-1])
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    width, height = pil_img.size
    rgb_array = np.array(pil_img)

    # Convert to PyTorch float tensor [C, H, W] in range [0.0, 1.0]
    tensor = torchvision.transforms.functional.to_tensor(pil_img)
    return tensor, (width, height), rgb_array


def filter_detections(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    categories: List[str],
    img_width: int,
    img_height: int,
    conf_threshold: float = 0.5,
) -> List[Detection]:
    """Filter raw model predictions by allowed classes, confidence, and boundary validity."""
    results: List[Detection] = []
    now_utc = datetime.now(timezone.utc)

    for box, label_idx, score in zip(boxes, labels, scores):
        conf = float(score.item())
        if conf < conf_threshold:
            continue

        category_idx = int(label_idx.item())
        if category_idx >= len(categories):
            continue
        class_name = categories[category_idx]

        if class_name not in ALLOWED_CLASSES:
            continue

        xmin, ymin, xmax, ymax = [float(coord.item()) for coord in box]

        # Reject boxes outside image boundaries or geometrically invalid
        if xmin < 0.0 or ymin < 0.0 or xmax > float(img_width) or ymax > float(img_height):
            logger.debug("Rejected out-of-bounds box: [%.1f, %.1f, %.1f, %.1f] for image (%d, %d)",
                         xmin, ymin, xmax, ymax, img_width, img_height)
            continue

        if xmin >= xmax or ymin >= ymax:
            logger.debug("Rejected degenerate box: [%.1f, %.1f, %.1f, %.1f]", xmin, ymin, xmax, ymax)
            continue

        # Create Detection schema instance with REAL data_source and real model confidence
        det = Detection(
            detection_id=f"det-{uuid.uuid4().hex[:8]}",
            label=class_name,
            bbox=[round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
            confidence=round(conf, 4),
            data_source="REAL",
            timestamp=now_utc,
        )
        results.append(det)

    return results


def detect(
    image: Union[str, Path, np.ndarray, Image.Image],
    conf_threshold: float = 0.5,
) -> List[Detection]:
    """Detect vehicles and persons in the provided image using Faster R-CNN ResNet-50 FPN.

    Args:
        image: Path to image file, PIL Image, or NumPy array.
        conf_threshold: Minimum confidence threshold [0.0 - 1.0]. Default 0.5.

    Returns:
        List of Detection schema objects (data_source="REAL", real confidence).
    """
    model, device, categories = get_detector_model()
    tensor, (width, height), _ = preprocess_image(image)

    # Send tensor to active device
    input_tensor = [tensor.to(device)]

    try:
        with torch.no_grad():
            outputs = model(input_tensor)
    except Exception as e:
        if device.type == "mps":
            print(f"[Third-Eye CV] Inference failed on MPS ({e}). Retrying on CPU fallback...")
            cpu_device = torch.device("cpu")
            model.to(cpu_device)
            input_tensor = [tensor.to(cpu_device)]
            with torch.no_grad():
                outputs = model(input_tensor)
        else:
            raise e

    output = outputs[0]
    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()

    return filter_detections(
        boxes=boxes,
        labels=labels,
        scores=scores,
        categories=categories,
        img_width=width,
        img_height=height,
        conf_threshold=conf_threshold,
    )


def detect_with_orientation(
    image: Union[str, Path, np.ndarray, Image.Image],
    conf_threshold: float = 0.5,
    min_rotation_confidence: float = 0.7,
) -> Tuple[List[Detection], Image.Image, dict]:
    """Detect objects, retrying quarter turns only when the source has none.

    Some real photos are stored physically rotated with no EXIF orientation tag.
    Faster R-CNN is not rotation invariant, so a frame with zero accepted
    detections is retried at 90/180/270 degrees. A rotated result is accepted
    only when it contains a detection at or above ``min_rotation_confidence``.
    The returned PIL image is the exact orientation used for inference, so all
    returned boxes are in that displayed image's pixel coordinates. The source
    file/bytes are never modified.
    """
    if isinstance(image, (str, Path)):
        pil_img = ImageOps.exif_transpose(Image.open(str(image))).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = ImageOps.exif_transpose(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image array, got shape {image.shape}")
        array = (
            image[:, :, ::-1]
            if image.dtype == np.uint8
            else (image * 255).astype(np.uint8)[:, :, ::-1]
        )
        pil_img = Image.fromarray(array).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    source_size = list(pil_img.size)
    info: dict = {
        "rotation_applied": False,
        "rotation_degrees": 0,
        "source_size_px": source_size,
        "analyze_size_px": source_size,
        "attempted_rotations_deg": [],
        "reason": "source orientation retained; detections found at original orientation",
    }

    detections = detect(pil_img, conf_threshold=conf_threshold)
    if detections:
        return detections, pil_img, info

    info["reason"] = (
        "no accepted detection at source orientation; retried 90/180/270 degrees "
        "and kept the best result at or above the rotation confidence floor"
    )
    best_detections: List[Detection] = []
    best_image = pil_img
    best_angle = 0
    best_confidence = 0.0

    for angle in (90, 180, 270):
        info["attempted_rotations_deg"].append(angle)
        rotated = pil_img.rotate(angle, expand=True)
        candidates = detect(rotated, conf_threshold=conf_threshold)
        top_confidence = max(
            (float(item.confidence or 0.0) for item in candidates), default=0.0
        )
        if candidates and top_confidence >= min_rotation_confidence and top_confidence > best_confidence:
            best_detections, best_image, best_angle = candidates, rotated, angle
            best_confidence = top_confidence

    if best_detections:
        info.update(
            rotation_applied=True,
            rotation_degrees=best_angle,
            analyze_size_px=list(best_image.size),
            best_rotation_confidence=round(best_confidence, 4),
            reason=(
                f"source had zero accepted detections; {best_angle} degree rotation "
                f"produced {len(best_detections)} detection(s) with top confidence "
                f"{best_confidence:.4f}; source file unchanged"
            ),
        )
    else:
        info["reason"] = (
            "no accepted detection at source orientation or at any quarter turn; "
            "no orientation was guessed"
        )
    return best_detections, best_image, info
