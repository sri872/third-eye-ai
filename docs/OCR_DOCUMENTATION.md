# Classical OCR Documentation — Indian Number Plate Recognition

## 1. Overview
The OCR module performs character recognition on Indian vehicle license plates without external neural engines (such as EasyOCR or Tesseract). It employs classical morphological plate localization, Hough line deskewing, connected-component character segmentation, 24x32 standardized template correlation, and statutory Indian registration validation.

---

## 2. Plate Preprocessing Pipeline

```text
Candidate Plate Crop (BGR)
   │
   ├─► Standardized Height Rescaling (Target height: 64 px, proportional width)
   │
   ├─► Hough Transform Deskewing (Detects line angles within ±25° to straighten plate horizontally)
   │
   ├─► Bilateral Filtering (Smoothes noise while retaining character stroke edges)
   │
   ├─► CLAHE Luminance Enhancement (Contrast-limited adaptive histogram equalization)
   │
   └─► Adaptive Otsu Binarization (Inversion logic ensures characters are white=255 on black=0)
```

---

## 3. Character Segmentation
- Uses connected-component / contour analysis (`cv2.RETR_CCOMP`) to extract character components located within plate frames.
- **Aspect ratio constraint**: Height/width between 0.70 and 5.5.
- **Relative height constraint**: Character height between 25% and 95% of plate height.
- **Nested component filtering**: Rejects small interior counters or noise completely enclosed inside character strokes.
- **Spatial sorting**: Left-to-right ordering by horizontal coordinate $x$.
- **Normalization**: Each segmented glyph is normalized with a proportional border to fixed dimensions $(24, 32)$ binary bitmap.

---

## 4. Standardized Template Matching & Positional Priors
Standardized binary templates $(24 \times 32)$ are generated for:
- 26 uppercase Latin letters: `A` to `Z`
- 10 decimal digits: `0` to `9`

### Match Scoring Function
For input character bitmap $C$ and template $T$:
$$\text{Score}(C, T) = 0.65 \cdot \text{NormCorr}(C, T) + 0.35 \cdot \text{IoU}(C, T)$$
where $\text{NormCorr}$ is the mapped normalized cross-correlation from `cv2.matchTemplate(TM_CCOEFF_NORMED)` and $\text{IoU}$ is the binary Jaccard intersection over union.

### Positional Priors for Indian Plates
When candidate length is ~10 characters:
- **Index 0, 1 (State Code)**: Enforce `letter` prior (evaluated strictly against `A-Z`).
- **Index 2, 3 (RTO District)**: Enforce `digit` prior (evaluated strictly against `0-9`).
- **Index 4, 5 (Series)**: Enforce `letter` prior (evaluated strictly against `A-Z`).
- **Index 6, 7, 8, 9 (Serial)**: Enforce `digit` prior (evaluated strictly against `0-9`).

This eliminates standard optical ambiguities (e.g. `0` vs `O`, `1` vs `I`, `8` vs `B`, `5` vs `S`).

---

## 5. Indian Registration Plate Validation
Validates candidate strings against statutory patterns:
1. **Standard Indian Series**: `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$`
   - Validates state code against 36 official RTO jurisdictions (e.g. `TS` -> Telangana, `KA` -> Karnataka, `DL` -> Delhi, `MH` -> Maharashtra, `PB` -> Punjab).
2. **Bharat Series**: `^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$`

### Verification Status Classes
- **`VERIFIED`**: Full registration matches statutory Indian regex AND mean template similarity $\ge 0.70$.
- **`UNVERIFIED`**: Partial regex match or similarity between $0.50$ and $0.70$.
- **`UNREADABLE`**: Low template similarity ($< 0.50$), broken characters, or non-matching syntax.
