"""Classical Computer Vision Image Processing Module.

Provides deterministic image loading, validation, enhancement, filtering,
and diagnostic operations using OpenCV and NumPy without any learned models.
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


def load_image(source: Any, rotation_degrees: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate an image from file path or raw bytes.

    Returns:
        Tuple of (image_bgr, image_rgb).

    Raises:
        ValueError: If image cannot be read or is empty.
    """
    if isinstance(source, (str, bytes, bytearray)):
        if isinstance(source, str):
            image = cv2.imread(source)
        else:
            nparr = np.frombuffer(source, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif hasattr(source, "read"):
        source.seek(0)
        nparr = np.frombuffer(source.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(source, np.ndarray):
        image = source.copy()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")

    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Decoded image is empty or invalid.")

    # Do not infer orientation from aspect ratio: valid portrait photos must stay portrait.
    # OpenCV applies supported EXIF orientation metadata while decoding image files.
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotation_degrees not in (0, 90, 180, 270):
        raise ValueError("rotation_degrees must be one of 0, 90, 180, or 270.")
    if rotation_degrees:
        image = cv2.rotate(image, rotations[rotation_degrees])

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, rgb


def enhance_image(
    image: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0,
    apply_clahe: bool = False,
    blur_type: str = "none",
    blur_kernel: int = 3,
) -> np.ndarray:
    """Apply classical enhancement operations to a BGR image.

    Args:
        image: Input BGR image.
        brightness: Additive brightness offset (-100 to 100).
        contrast: Multiplicative contrast factor (0.5 to 3.0).
        apply_clahe: Whether to apply CLAHE to L-channel of LAB.
        blur_type: 'none', 'gaussian', or 'median'.
        blur_kernel: Odd integer kernel size (e.g. 3, 5).

    Returns:
        Enhanced BGR image.
    """
    out = image.astype(np.float32)

    # Contrast and brightness: out = alpha * image + beta
    if contrast != 1.0 or brightness != 0:
        out = contrast * out + brightness
        out = np.clip(out, 0, 255)

    out = out.astype(np.uint8)

    # Optional CLAHE on LAB luminance channel
    if apply_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Blur filtering
    kernel_size = max(3, blur_kernel)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if blur_type == "gaussian":
        out = cv2.GaussianBlur(out, (kernel_size, kernel_size), 0)
    elif blur_type == "median":
        out = cv2.medianBlur(out, kernel_size)

    return out


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_edges(
    gray: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    aperture_size: int = 3,
) -> np.ndarray:
    """Detect edges using Canny edge detector."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low_thresh, high_thresh, apertureSize=aperture_size)


def compute_image_diagnostics(image: np.ndarray) -> Dict[str, Any]:
    """Compute classical image quality and statistical diagnostics."""
    if image is None or image.size == 0:
        return {"valid": False, "error": "Empty image"}

    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    gray = to_grayscale(image)

    # Mean brightness and contrast
    mean_val, std_val = cv2.meanStdDev(gray)
    brightness = float(mean_val[0][0])
    contrast = float(std_val[0][0])

    # Blur metric: variance of the Laplacian (higher = sharper, < 100 often blurry)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Lighting classification
    if brightness < 60:
        lighting = "LOW_EXPOSURE"
    elif brightness > 195:
        lighting = "HIGH_EXPOSURE_OR_GLARE"
    else:
        lighting = "MID_RANGE_EXPOSURE"

    # Sharpness classification
    if sharpness < 80:
        clarity = "BLURRED_OR_DEFOCUSED"
    elif sharpness < 200:
        clarity = "MODERATE_SHARPNESS"
    else:
        clarity = "HIGH_SHARPNESS"

    return {
        "valid": True,
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "mean_brightness": round(brightness, 2),
        "contrast_std": round(contrast, 2),
        "sharpness_laplacian_var": round(sharpness, 2),
        "exposure_assessment": lighting,
        "clarity_assessment": clarity,
    }
