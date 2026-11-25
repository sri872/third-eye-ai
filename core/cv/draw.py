"""Visualization utility for computer vision detections in Project Third-Eye.

Draws bounding boxes and class labels with real model confidence scores using OpenCV.
Saves annotated images to data/outputs/.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from core.schemas import Detection

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "outputs"

# Color palette (BGR format for OpenCV)
CLASS_COLORS = {
    "car": (0, 255, 0),         # Vibrant Green
    "truck": (0, 165, 255),      # Orange
    "bus": (255, 191, 0),        # Deep Sky Blue
    "motorcycle": (255, 0, 255), # Magenta
    "bicycle": (0, 255, 255),    # Yellow
    "person": (0, 0, 255),       # Red
}
DEFAULT_COLOR = (0, 255, 0)


def draw_detections(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    detections: List[Detection],
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, Path]:
    """Draw bounding boxes and confidence labels on an image and save to disk.

    Args:
        image_input: Path to source image, NumPy array, or PIL Image.
        detections: List of Detection schema objects to render.
        output_path: Optional destination file path. If None, saves to data/outputs/<stem>_annotated.jpg.

    Returns:
        annotated_img: NumPy image array (BGR format).
        saved_path: Path where the annotated image was saved.
    """
    stem = "image"
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image from path: {img_path}")
    elif isinstance(image_input, Image.Image):
        # Convert PIL to BGR NumPy array
        rgb = np.array(image_input.convert("RGB"))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Ensure output directory exists
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        target_path = DEFAULT_OUTPUT_DIR / f"{stem}_annotated.jpg"
    else:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw each detection box and text label
    for det in detections:
        xmin, ymin, xmax, ymax = [int(round(c)) for c in det.bbox]
        color = CLASS_COLORS.get(det.label, DEFAULT_COLOR)

        # Draw box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        # Construct label text with real model confidence
        if det.confidence is not None:
            label_text = f"{det.label}: {det.confidence:.2f}"
        else:
            label_text = det.label

        # Compute label size and draw background box for legibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        text_ymin = max(ymin, text_h + 4)
        cv2.rectangle(
            img,
            (xmin, text_ymin - text_h - 4),
            (xmin + text_w + 4, text_ymin + baseline - 2),
            color,
            cv2.FILLED,
        )
        # Put text in high-contrast black or white
        text_color = (0, 0, 0)
        cv2.putText(
            img,
            label_text,
            (xmin + 2, text_ymin - 2),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    # Save to target path
    success = cv2.imwrite(str(target_path), img)
    if not success:
        raise IOError(f"Failed to save annotated image to {target_path}")

    return img, target_path
