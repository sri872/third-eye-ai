# Classical Computer Vision Documentation

## 1. Purpose and Philosophy
The Computer Vision module provides deterministic, explainable image processing and candidate region localization. Instead of relying on opaque deep neural networks, it utilizes classical morphological operations and edge statistics to identify candidate vehicle and plate regions.

---

## 2. Pipeline Architecture

```text
Input Image (BGR)
   │
   ├─► Orientation Correction (Height vs Width check)
   │
   ├─► Image Diagnostics (Mean brightness, std contrast, Laplacian blur metric)
   │
   ├─► Enhancement Pipeline (Brightness, contrast scaling, optional CLAHE, Bilateral/Gaussian smoothing)
   │
   ├─► Canny Edge Detection (Configurable hysteresis thresholds: default 40, 130)
   │
   ├─► Horizontal Morphology (Closing with 19x5 rectangular structuring element)
   │
   ├─► Contour Analysis (Area ratio filtering: 1.5% - 85%; Aspect ratio filtering: 0.5 - 3.8)
   │
   ├─► Edge Density Evaluation (Rejects low-texture background noise)
   │
   └─► Non-Maximum Suppression (IoU threshold 0.45) ──► Candidate Vehicle Bounding Boxes
```

---

## 3. Vehicle Candidate Region Localization
In road images, vehicles exhibit strong structural characteristics:
- **Prominent horizontal boundaries**: Bumpers, rooflines, windshield bases, and road contact shadows.
- **Aspect ratio range**: Typically 0.8 to 2.5 for passenger cars, and up to 3.5 for trucks/buses.
- **Edge concentration**: Vehicles contain higher edge densities (grille, windows, wheels) compared to plain asphalt or sky.

Candidate regions are flagged with an explicit label:
`GEOMETRIC CANDIDATE (CLASSICAL CV HEURISTIC)`
and assigned a geometric confidence metric based on area, aspect ratio fit, and internal edge density.

---

## 4. Quality Diagnostics
Image metrics computed:
- **Dimensions**: Width × Height × Channels.
- **Mean Brightness**: Grayscale luminance average (0–255).
- **Contrast**: Grayscale standard deviation.
- **Sharpness Metric**: Variance of the Laplacian ($\sigma^2_{Lap}$).
  - $> 200$: High sharpness
  - $80 - 200$: Moderate clarity
  - $< 80$: Blurred or defocused
- **Lighting Assessment**: `NORMAL_DAYLIGHT`, `LOW_LIGHT_OR_NIGHT`, or `HIGH_GLARE_OR_OVEREXPOSED`.

---

## 5. Limitations
1. **Geometric proposals only**: A candidate region indicates an edge and contour cluster satisfying vehicle geometry; it does not confirm vehicle presence.
2. **Cluttered backgrounds**: Roadside buildings, signboards, or dense tree canopies with strong edges may produce candidate false positives.
3. **Occlusions**: Heavily overlapping vehicles in dense traffic may merge into a single candidate bounding box.
