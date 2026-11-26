"""Multi-path classical number-plate candidate localization.

This module proposes plate-shaped image regions using OpenCV morphology,
gradients, adaptive thresholds and geometric filters. Proposals are not
confirmed plates; OCR/format checks must rank them separately.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _iou(a: List[int], b: List[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union else 0.0


def _candidate_score(box: Tuple[int, int, int, int], edge_map: np.ndarray) -> tuple[float, float, float]:
    x, y, w, h = box
    ar = w / float(max(1, h))
    crop = edge_map[y:y + h, x:x + w]
    density = float(cv2.countNonZero(crop)) / float(max(1, w * h))
    # Indian registration plates are commonly wide rectangles. This is a
    # ranking prior only; it is not a hard format detector.
    ratio_fit = max(0.0, 1.0 - abs(ar - 4.0) / 3.0)
    edge_fit = min(1.0, density * 4.0)
    horizontal_fit = min(1.0, (w / float(max(1, h))) / 4.0)
    score = 0.50 * ratio_fit + 0.35 * edge_fit + 0.15 * horizontal_fit
    return round(score, 4), round(ar, 3), round(density, 4)


def _deduplicate(candidates: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for candidate in ordered:
        if all(_iou(candidate["bbox"], prior["bbox"]) < iou_threshold for prior in kept):
            kept.append(candidate)
    return kept


def localize_plate_candidates(
    image: np.ndarray,
    search_regions: Optional[List[Tuple[int, int, int, int]]] = None,
    min_aspect_ratio: float = 1.8,
    max_aspect_ratio: float = 6.5,
    min_area: int = 250,
    max_area: int = 250000,
) -> List[Dict[str, Any]]:
    """Return plate-like candidate crops using several classical CV pathways.

    Search regions are additional ROIs, not a replacement for full-frame
    search. Callers that need both should invoke this function for each set
    and merge the results.
    """
    if image is None or image.size == 0:
        return []
    if image.ndim not in (2, 3):
        return []

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    regions = [(0, 0, w_img, h_img)]
    if search_regions:
        regions.extend(search_regions)

    candidates: List[Dict[str, Any]] = []
    seen_regions = set()

    for rx, ry, rw, rh in regions:
        rx = max(0, min(int(rx), w_img - 1))
        ry = max(0, min(int(ry), h_img - 1))
        rw = max(1, min(int(rw), w_img - rx))
        rh = max(1, min(int(rh), h_img - ry))
        region_key = (rx, ry, rw, rh)
        if region_key in seen_regions:
            continue
        seen_regions.add(region_key)

        roi = gray[ry:ry + rh, rx:rx + rw]
        if roi.shape[0] < 18 or roi.shape[1] < 36:
            continue

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        grad_x = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
        edge_map = cv2.Canny(blurred, 40, 140)

        # Multiple pathways reduce dependence on a single lighting/polarity
        # assumption. Kernels scale with the ROI instead of assuming one plate size.
        close_w = max(11, min(31, int(round(rw * 0.055))))
        if close_w % 2 == 0:
            close_w += 1
        blackhat_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(9, min(29, int(round(rw * 0.07))) | 1), max(3, min(9, int(round(rh * 0.12))) | 1)),
        )
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, blackhat_kernel)

        paths: List[np.ndarray] = []
        for source in (grad_x, cv2.add(grad_x, blackhat), cv2.add(grad_x, tophat)):
            source_blur = cv2.GaussianBlur(source, (5, 5), 0)
            _, otsu = cv2.threshold(source_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            paths.append(otsu)

        for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            adaptive = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, polarity,
                25 if min(roi.shape) >= 25 else  fifteen_odd(min(roi.shape)), 7
            )
            paths.append(adaptive)

        for binary in paths:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 3))
            merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            merged = cv2.morphologyEx(
                merged, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            )
            contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                px, py, pw, ph = cv2.boundingRect(contour)
                area = pw * ph
                if area < min_area or area > max_area:
                    continue
                if pw < 28 or ph < 9:
                    continue
                ar = pw / float(max(1, ph))
                if not min_aspect_ratio <= ar <= max_aspect_ratio:
                    continue

                # Expand the crop slightly so outer plate borders and edge
                # characters are not cut off by the morphology contour.
                pad_x = max(2, int(round(pw * 0.05)))
                pad_y = max(2, int(round(ph * 0.12)))
                x1 = max(0, rx + px - pad_x)
                y1 = max(0, ry + py - pad_y)
                x2 = min(w_img, rx + px + pw + pad_x)
                y2 = min(h_img, ry + py + ph + pad_y)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                local_edges = edge_map[py:py + ph, px:px + pw]
                score, measured_ar, density = _candidate_score((px, py, pw, ph), edge_map)
                candidates.append({
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "contour_bbox": [int(rx + px), int(ry + py), int(pw), int(ph)],
                    "aspect_ratio": round(measured_ar, 2),
                    "area": int((x2 - x1) * (y2 - y1)),
                    "edge_density": density,
                    "score": score,
                    "crop": crop,
                    "method": "MULTI_PATH_CLASSICAL_MORPHOLOGY",
                    "search_region": list(region_key),
                })

    return _deduplicate(candidates)[:30]


def fifteen_odd(limit: int) -> int:
    """Return a legal adaptive-threshold block size for very small ROIs."""
    value = max(3, limit - 1)
    return value if value % 2 else value - 1
