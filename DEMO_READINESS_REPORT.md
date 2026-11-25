# THIRD EYE AI — Demonstration Readiness Report

> **Historical report.** This document records the mid-project readiness audit. Some OCR values and test counts were superseded by later fixes. The authoritative final measurements are in [FINAL_DEMO_VERIFICATION_REPORT.md](FINAL_DEMO_VERIFICATION_REPORT.md) and the machine-readable evidence files under `data/outputs/`.

**Workspace:** `~/Projects/third-eye-v2`  
**Report date:** 2026-11-25  
**Baseline suite:** `204 passed, 6 warnings in 38.38s`  
**Final suite:** `218 passed, 11 warnings in 90.22s (0:01:30)` — 0 failed  
**Targeted suite:** `69 passed, 10 warnings in 59.87s` — 0 failed  
**Overall status:** **PARTIAL — all mandatory workflows run with real outputs; two model limitations remain and are disclosed below.**

This report replaces the previous stale report. Every number below came from an actual run on the four mandatory inputs. No fabricated boxes, scores, OCR text, calls, or dispatch actions are included.

---

## 1. Acceptance table

| # | Acceptance criterion | Status | Measured evidence |
|---|---|---|---|
| 1 | All four mandatory inputs exist, decode, and preserve originals | **PASS** | Four files present; SHA-256 hashes unchanged after all runs; 3 images decoded; all 525 video frames decoded. See §2. |
| 2 | Faster R-CNN runs on all three images with real boxes, labels, confidences, and counts | **PASS** | 1 / 9 / 45 real detections on the three images; counts and exact boxes in §4. Zero-detection photo recovered by 90° analysis rotation. |
| 3 | Plate localisation → crop → EasyOCR → raw evidence → threshold verification | **PARTIAL** | Real YuNet candidate, real EasyOCR raw evidence, and strict 0.7 gating all ran. `TS07JA0566` candidate reached only **0.4348**, so it stays **unverified**; no verified registration number is claimed. See §5. |
| 4 | Three exact caller descriptions are processed without adding unstated facts | **PASS** | Exact texts parsed; `NH-44`, 2 cars, 1 person, `injured`, and `emergency assistance` extracted; vague `many`/`several` left non-numeric; no invented injuries or urgency. See §6. |
| 5 | Video processed through sampled frames with timestamps, boxes, scores, and temporal aggregation | **PASS** | 12 real frames at `0.0`–`20.96 s`; 17 boxes on the top frame; 0.4201 peak; temporal signal `ISOLATED UNCERTAIN FRAME IN SAMPLE`. Sampling disclaimer shown in UI. See §8. |
| 6 | Streamlit exposes all six tabs and each mandatory workflow | **PASS** | Six exact tab labels verified in browser; images verified in browser; video verified through Streamlit `AppTest` because the browser file tool has a 5 MiB cap. No console errors or AppTest exceptions. See §9. |
| 7 | No stale cached detections; unreadable/unsupported media handled gracefully | **PASS** | Cache key includes filename, byte length, SHA-256, and sample count; corrupt `broken.jpg` produced a clear error with zero exceptions. See §9.3. |
| 8 | Targeted + full pytest suites preserve baseline | **PASS** | Targeted: 69 passed. Full: 218 passed vs 204 baseline. See §11. |
| **Overall** | **Demonstration readiness** | **PARTIAL** | Ready to demonstrate with honest disclosures; not a claim of verified plate text, accident confirmation, or full-clip video inference. |

---

## 2. Mandatory inputs — existence, decode validity, dimensions, EXIF, and integrity

All originals remain under `data/samples/demo/`. No original media file or database record was modified.

| Input | Size (bytes) | File dimensions (W×H) | Decoded array | EXIF orientation | SHA-256 |
|---|---:|---:|---:|---|---|
| `single_car_plate.jpg` | 114,036 | 900×1600 | 1600×900×3 | none | `246cf1cd4dc1cdc58592c4b92878289c70a71c051039d3c71e4a417322fb612b` |
| `two_car_collision.jpg` | 312,179 | 1771×1237 | 1237×1771×3 | none | `3fdb9739daeba222e2782aabf02952577ee3a3dc6a81b3a6ba7883e02009c3c9` |
| `heavy_traffic.jpg` | 172,786 | 885×1270 | 1270×885×3 | none | `eb25eb9229002de506d4c622bd43793c461b6ddeb51afb5f4605d25be77967b9` |
| `road vdo.mp4` | 14,323,936 | 1920×1080 | 25.0 FPS, 525 frames, 21.0 s | n/a | `cdc01a90b62616b62abeeaa8110de9314a60c4dce0d140820cbd7d54c7a76809` |

Video decode check: **525/525 frames decoded successfully**. The hashes above were recomputed after all analysis and match the values recorded before analysis.

YuNet model integrity: `models/plate_detector/license_plate_detection_lpd_yunet_2023mar.onnx` hash is
`6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7`, matching the pinned constant; `verify_model_file()` returned `True`.

---

## 3. Defects investigated and corrected

1. **Physically rotated image with no EXIF tag.** `single_car_plate.jpg` is stored as a sideways 900×1600 photo with no EXIF orientation. Faster R-CNN returned zero accepted detections at source orientation. `detect_with_orientation()` retried 90/180/270° only after that zero-detection result, accepted the 90° result at 0.9199 confidence, and returned the displayed analysis image so all boxes are in displayed coordinates. The original file/bytes were not modified.
2. **EXIF handling.** `detect()` and `detect_with_orientation()` now apply `ImageOps.exif_transpose()` before inference, so tagged images are respected. This image had no tag, so the separate quarter-turn recovery path was required.
3. **Plate candidate threshold.** YuNet's candidate score on the Indian plate is about **0.27**, below a 0.50 proposal cutoff. Candidate proposal now uses **0.25**. This does **not** lower registration verification: `PLATE_TRUST_THRESHOLD` remains **0.70**.
4. **Whole-crop OCR fallback.** When YuNet and the deterministic heuristic both miss, a plate-like EasyOCR text box proposes a padded candidate region, which is cropped and re-read. The fallback filter requires a normalized candidate of at least 8 characters with at least 3 digits; this rejects false proposals such as the observed whole-crop read `can09g` at 0.5412.
5. **Raw-vs-preprocessed plate evidence.** The high-confidence raw read `IND` (0.9457) is retained verbatim as raw evidence, while the lower-confidence alphanumeric candidate `TS07JA0566` (0.4348) is selected as the plate candidate. No characters are auto-corrected.
6. **Caller extraction misses.** Added Indian `NH`/`SH` route matching, singular `one person`/`a person`, and the exact phrase `emergency assistance`. A generic `emergency` keyword was deliberately **not** added. Optional adjectives such as `damaged` are tolerated in vehicle-count patterns.
7. **Video sample-count control.** `_process_video()` previously had a default sample count that ignored the sidebar slider; it now accepts and clamps `sample_count`, and `app/main.py` passes the selected value.
8. **Stale cache risk.** The UI result key includes filename, byte length, content SHA-256, and `video_sample_count`; changing any of these invalidates cached detections.
9. **Unreadable/unsupported media.** Decode/inference failures are caught and displayed as a clear message instead of crashing the session; unreadable plates remain `NOT READABLE (UNVERIFIED)`.

---

## 4. Real Faster R-CNN results on the three images

Model: `Faster R-CNN ResNet-50 FPN`, COCO-pretrained, device **MPS**, detection confidence threshold **0.50**. Boxes are `[xmin, ymin, xmax, ymax]` in displayed analysis-image pixels.

### 4.1 Summary

| Image | Rotation applied | Detections | Class counts | Accident classifier score | Verdict | Still-image status |
|---|---:|---:|---|---:|---|---|
| `single_car_plate.jpg` | 90° | 1 | 1 car | 0.4230 | UNCERTAIN | `NOT DETERMINED (out-of-domain still image)` |
| `two_car_collision.jpg` | 0° | 9 | 7 cars, 2 persons | 0.2166 | NO ACCIDENT SIGNAL | `NOT DETERMINED (out-of-domain still image)` |
| `heavy_traffic.jpg` | 0° | 45 | 12 cars, 10 motorcycles, 23 persons | 0.3903 | NO ACCIDENT SIGNAL | `NOT DETERMINED (out-of-domain still image)` |

Geometric overlap context is explanatory only, not a verdict: collision had `max_vehicle_iou=0.1375`, `mean_vehicle_iou=0.0143`, and 7 heavily overlapping pairs among 21 pairs; heavy traffic had `max_vehicle_iou=0.4981` and 10 heavily overlapping pairs among 231 pairs.

### 4.2 Exact boxes — `single_car_plate.jpg`

| # | Label | Confidence | Box |
|---:|---|---:|---|
| 1 | car | 0.9199 | `[353.91, 119.53, 1274.37, 879.43]` |

### 4.3 Exact boxes — `two_car_collision.jpg`

| # | Label | Confidence | Box |
|---:|---|---:|---|
| 1 | car | 0.9992 | `[909.90, 541.20, 1720.27, 908.30]` |
| 2 | car | 0.9978 | `[218.39, 570.32, 966.71, 887.65]` |
| 3 | person | 0.9973 | `[1055.70, 507.41, 1153.82, 642.92]` |
| 4 | car | 0.9925 | `[909.94, 566.14, 988.88, 627.80]` |
| 5 | car | 0.9741 | `[823.48, 574.37, 874.98, 611.47]` |
| 6 | car | 0.9699 | `[977.05, 562.50, 1039.74, 608.06]` |
| 7 | person | 0.9426 | `[1218.17, 494.81, 1263.37, 549.58]` |
| 8 | car | 0.6837 | `[1025.94, 557.88, 1061.38, 600.87]` |
| 9 | car | 0.6038 | `[987.93, 538.84, 1020.77, 563.05]` |

### 4.4 Exact boxes — `heavy_traffic.jpg`

<details>
<summary>All 45 real detections (label, confidence, box)</summary>

```text
 1  person     0.9997  [672.99, 510.27, 838.00, 801.17]
 2  person     0.9997  [23.22, 436.10, 181.73, 720.04]
 3  person     0.9996  [264.77, 557.46, 418.16, 813.34]
 4  person     0.9983  [189.27, 411.36, 320.94, 691.90]
 5  motorcycle 0.9917  [896.37, 8.56, 990.31, 157.48]
 6  person     0.9882  [478.10, 319.89, 618.80, 566.32]
 7  person     0.9852  [1006.62, 321.42, 1155.30, 588.15]
 8  car        0.9846  [3.30, 57.54, 193.60, 271.55]
 9  motorcycle 0.9843  [612.39, 11.25, 689.19, 139.74]
10  car        0.9835  [56.15, 0.00, 274.88, 106.99]
11  person     0.9819  [914.75, 302.30, 1028.76, 520.46]
12  car        0.9692  [111.05, 193.31, 451.46, 515.44]
13  motorcycle 0.9646  [908.13, 368.88, 1024.07, 571.32]
14  car        0.9637  [716.19, 26.18, 919.71, 249.45]
15  car        0.9627  [0.24, 170.44, 131.10, 489.29]
16  car        0.9509  [254.22, 16.30, 506.78, 265.05]
17  person     0.9448  [1209.28, 163.89, 1270.00, 371.64]
18  car        0.9443  [630.10, 128.74, 902.75, 484.71]
19  person     0.9322  [470.41, 234.76, 560.26, 361.44]
20  motorcycle 0.9288  [1125.63, 344.51, 1236.72, 561.91]
21  car        0.9258  [705.37, 0.00, 884.44, 82.02]
22  person     0.9136  [1013.44, 153.36, 1091.85, 321.74]
23  car        0.9093  [995.12, 31.36, 1221.08, 268.02]
24  motorcycle 0.9022  [1053.93, 429.76, 1151.52, 600.35]
25  person     0.8996  [1122.39, 243.00, 1211.80, 365.90]
26  person     0.8764  [579.75, 81.26, 653.67, 231.11]
27  motorcycle 0.8744  [1215.42, 18.21, 1270.00, 131.78]
28  person     0.8661  [966.94, 190.76, 1047.44, 404.52]
29  person     0.8647  [531.60, 155.09, 629.37, 360.90]
30  person     0.8390  [199.22, 269.12, 262.99, 319.83]
31  person     0.8274  [525.85, 0.00, 597.78, 104.12]
32  motorcycle 0.8236  [492.05, 405.38, 599.33, 585.10]
33  motorcycle 0.7955  [527.37, 19.67, 596.43, 134.99]
34  person     0.7819  [21.81, 3.20, 48.42, 52.60]
35  person     0.7785  [1124.55, 248.72, 1243.77, 510.80]
36  car        0.7716  [978.68, 0.50, 1147.16, 59.32]
37  person     0.7385  [0.41, 16.02, 26.25, 63.93]
38  person     0.7229  [905.71, 0.00, 984.95, 121.61]
39  car        0.6995  [976.02, 4.11, 1138.37, 106.40]
40  person     0.6558  [677.38, 246.78, 737.29, 296.03]
41  car        0.6394  [308.78, 2.11, 523.71, 114.08]
42  person     0.6060  [992.98, 156.61, 1089.86, 423.45]
43  person     0.6034  [611.87, 0.00, 685.02, 104.26]
44  motorcycle 0.5971  [581.77, 142.26, 654.39, 241.38]
45  motorcycle 0.5061  [532.69, 192.96, 629.33, 370.84]
```

</details>

---

## 5. Plate localisation, EasyOCR evidence, and trust gating

`PLATE_TRUST_THRESHOLD = 0.70` was **not changed**. The UI never promotes a candidate below 0.70 to verified.

### 5.1 `single_car_plate.jpg`

- YuNet candidate detector score: **0.27**; displayed source-pixel bbox: `[628, 780, 949, 840]`.
- EasyOCR was run on the cropped region, both raw and contrast-enhanced.
- Raw evidence retained by the application: **`IND` at 0.9457**.
- Plate-like candidate selected from the alphanumeric read: **`TS07JA0566` at 0.4348**.
- A direct raw-crop diagnostic also observed `TS07 Ja0566` at **0.4348**; preprocessing variants reached 0.4537 and 0.4089. All are below 0.70.
- UI result: plate shown as `TS07JA0566 (unverified)`, OCR confidence `0.43`, plate status `unverified`. It is **not** a verified registration number.
- YuNet limitation: the vendored OpenCV Zoo model was trained on Chinese plates; it is used only as a candidate localiser, not as proof of text.

### 5.2 Other two images

| Image | Plate candidate regions | Real OCR outcome | UI plate status |
|---|---:|---|---|
| `two_car_collision.jpg` | 1 of 7 vehicles produced a YuNet candidate (`[988, 543, 1021, 556]`, score 0.26) | `NO OCR RESULTS`; no text invented | `NOT READABLE (UNVERIFIED)` |
| `heavy_traffic.jpg` | 2 of 22 vehicles produced YuNet candidates (`[612, 25, 689, 119]` at 0.33; `[1106, 229, 1142, 240]` at 0.44) | `NO OCR RESULTS` on those crops | `NOT READABLE (UNVERIFIED)` |

The UI states that a plate bbox is a candidate region, EasyOCR evidence is preserved verbatim, reads below 0.7 stay unverified, and missing reads are `NOT READABLE (UNVERIFIED)`.

---

## 6. Exact caller descriptions processed

The three descriptions were processed character-for-character. Sample call files `call_09_demo_collision.txt`, `call_10_demo_traffic.txt`, and `call_11_demo_uncertain.txt` contain these exact caller texts. They are labelled **SIMULATED SAMPLE CALL** and are arbitrarily paired with uploaded media.

| Scenario | Exact caller text | Extracted facts | Deliberately not inferred |
|---|---|---|---|
| Collision | `Two cars collided near NH-44. One person may be injured. Please send emergency assistance.` | location `NH-44`; vehicles `2`; types `car`; people `1`; injury keywords `[injured]`; urgency `[emergency assistance]` | No confirmed injury, no real dispatch, no accident override from the call |
| Traffic | `There are many cars and motorcycles near the main road. Several pedestrians are crossing. No collision has been reported.` | location `the main road`; types `[car, motorcycle]`; no injury/hazard/urgency keywords | `many` and `several` were not converted into numeric counts; negation kept |
| Uncertain | `I saw a vehicle stopped on the road. I am not sure whether an accident happened.` | location `the road`; vehicles `1`; no injury/hazard/urgency keywords | No accident or injury was invented from uncertainty |

`data/sample_calls/expected.json` was not changed; the benchmark still evaluates **8** files. Additional `call_demo_*.txt` files exist as UI options but are not part of that answer key.

---

## 7. Source-separated fusion results

Fusion keeps REAL vision and SIMULATED caller evidence in separate, labelled fields. The REAL vision verdict remains authoritative for uploaded media; simulated caller urgency never overrides it.

| Paired scenario | Vision verdict / score | Caller evidence | Fusion outcome |
|---|---|---|---|
| `two_car_collision.jpg` + collision call | NO ACCIDENT SIGNAL / 0.2166 | NH-44, 2 vehicles, 1 person, injured, emergency assistance | Count disagreements recorded: vision 7 vehicles vs caller 2; vision 2 visible people vs caller 1. No silent merge; dispatcher verification required. |
| `heavy_traffic.jpg` + traffic call | NO ACCIDENT SIGNAL / 0.3903 | `the main road`; car + motorcycle types; no numeric count, injury, hazard, or urgency | No disagreement. Caller types align with real detected classes; no accident claimed. |
| `single_car_plate.jpg` + uncertain call | UNCERTAIN / 0.4230 | `the road`; 1 vehicle; no injury, hazard, or urgency | No count disagreement; uncertainty retained; unknown injuries remain `MISSING`. |

Note: the collision classifier score is low because the classifier was trained on forward-facing dashcam video frames, while these are still images from other viewpoints. The UI explicitly says normal traffic cannot be confirmed from these stills and human review is required.

---

## 8. Video D — `road vdo.mp4`

### 8.1 Metadata and sampling

- 1920×1080, 25.0 FPS, 525 frames, 21.0 s; all 525 frames decoded.
- **12 evenly spaced frames** were sampled. This is an even-sample review, **not full-clip inference**; the UI shows this disclaimer.
- The displayed scene image is the highest-scoring sampled frame: **source frame 381 at 15.24 s**.
- OCR scope is limited to that highest-scoring frame, because running OCR on every frame is too slow; this is disclosed in Tab 2.

### 8.2 Real per-frame classifier scores

| Sample | Source frame | Timestamp (s) | Score | Verdict | Vehicles | Persons |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 0 | 0.00 | 0.2378 | NO ACCIDENT SIGNAL | 15 | 3 |
| 2 | 48 | 1.92 | 0.2854 | NO ACCIDENT SIGNAL | 15 | 3 |
| 3 | 95 | 3.80 | 0.2643 | NO ACCIDENT SIGNAL | 22 | 5 |
| 4 | 143 | 5.72 | 0.1915 | NO ACCIDENT SIGNAL | 14 | 9 |
| 5 | 191 | 7.64 | 0.3937 | NO ACCIDENT SIGNAL | 23 | 1 |
| 6 | 238 | 9.52 | 0.3148 | NO ACCIDENT SIGNAL | 15 | 4 |
| 7 | 286 | 11.44 | 0.1915 | NO ACCIDENT SIGNAL | 12 | 6 |
| 8 | 333 | 13.32 | 0.2147 | NO ACCIDENT SIGNAL | 16 | 5 |
| 9 | 381 | 15.24 | **0.4201** | **UNCERTAIN** | 14 | 3 |
| 10 | 429 | 17.16 | 0.3929 | NO ACCIDENT SIGNAL | 14 | 5 |
| 11 | 476 | 19.04 | 0.2657 | NO ACCIDENT SIGNAL | 14 | 6 |
| 12 | 524 | 20.96 | 0.2741 | NO ACCIDENT SIGNAL | 17 | 6 |

### 8.3 Boxes on the highest-scoring frame

Frame 381 produced 17 real detections: 12 cars, 1 truck, 1 motorcycle, 3 persons. Overall video verdict is **UNCERTAIN** with uncalibrated score **0.4201**.

<details>
<summary>All 17 boxes on source frame 381</summary>

```text
 1  motorcycle 0.9955  [202.22, 541.08, 318.12, 713.70]
 2  person     0.9946  [1700.20, 612.34, 1739.85, 745.68]
 3  person     0.9905  [211.89, 480.71, 317.13, 687.99]
 4  car        0.9894  [593.10, 516.64, 934.51, 732.00]
 5  car        0.9866  [0.22, 572.28, 113.98, 671.46]
 6  car        0.9763  [471.86, 606.89, 594.74, 672.84]
 7  car        0.9616  [1486.41, 600.84, 1664.33, 747.35]
 8  car        0.9577  [469.77, 608.85, 533.16, 676.37]
 9  car        0.9502  [125.68, 573.65, 221.30, 671.37]
10  car        0.9471  [431.32, 604.51, 503.53, 676.21]
11  car        0.9199  [1169.73, 591.60, 1285.30, 721.81]
12  car        0.8896  [915.96, 528.73, 1192.09, 735.03]
13  car        0.7915  [330.14, 595.29, 454.43, 678.95]
14  car        0.7903  [0.00, 578.32, 18.93, 669.76]
15  car        0.7311  [1647.02, 625.29, 1710.47, 725.99]
16  person     0.5371  [1018.02, 561.46, 1060.59, 590.91]
17  truck      0.5370  [922.69, 526.85, 1215.55, 735.26]
```

</details>

### 8.4 Temporal aggregation

| Metric | Measured value |
|---|---:|
| Sampled frames | 12 |
| Min / mean / max score | 0.1915 / 0.2872 / 0.4201 |
| Scores ≥0.60 (high) | 0 |
| Scores 0.40–0.60 (uncertain) | 1 |
| Scores <0.40 (no signal) | 11 |
| Longest run ≥0.40 | 1 sampled frame |
| Peak | frame 381 at 15.24 s |
| Temporal signal | `ISOLATED UNCERTAIN FRAME IN SAMPLE` |

The UI explicitly states that temporal evidence is explanatory only and does not override the highest-scoring frame verdict. No accident is confirmed from this sample.

---

## 9. Streamlit UI verification

### 9.1 Six tabs

The running app exposed exactly these six labels:

1. `1. Scene`
2. `2. Vehicles`
3. `3. Accident check`
4. `4. Incident report`
5. `5. Hospitals & emergency numbers`
6. `6. Dispatch`

All six were opened and read for the image workflows. No browser console errors were observed.

### 9.2 Browser image workflows

Verification server: `http://localhost:8504`.

| Upload | Paired call | Observed UI evidence |
|---|---|---|
| `two_car_collision.jpg` | `call_09_demo_collision.txt` | Fingerprint `3fdb97…09c3c9`; `source orientation retained`; 7 vehicles, 2 persons; score 0.2166; verdict NO ACCIDENT SIGNAL; `ARBITRARY DEMO PAIRING`. |
| `single_car_plate.jpg` | `call_11_demo_uncertain.txt` | Fingerprint `246cf1…612b`; `90 degree rotation applied (source had zero accepted detections; 90 degree rotation produced 1 detection(s) with top confidence 0.9199; source file unchanged)`; 1 vehicle, 0 persons; score 0.4230; plate `TS07JA0566 (unverified)`, 0.43, status `unverified`. |
| `heavy_traffic.jpg` | `call_10_demo_traffic.txt` | Fingerprint `eb25eb…967b9`; no rotation; 22 vehicles, 23 persons; score 0.3903; unreadable plates shown as `NOT READABLE (UNVERIFIED)`; all six tabs rendered without errors. |

The Tab 4 report for each image clearly separated `REAL (VISION)` from `SIMULATED (CALLER)` and `PROVENANCE METADATA`, and labelled the pairing `ARBITRARY DEMO PAIRING`.

### 9.3 Video, cache, and graceful-failure checks

- The browser file-upload tool has a **5 MiB total cap**, while `road vdo.mp4` is 14.3 MB. This is a verification-tool limitation, not an app limit (the app uploader allows 200 MB per file). The full video workflow was therefore exercised with Streamlit `AppTest`.
- AppTest result: six tabs; `Media fingerprint (SHA-256)` present; `uploaded video (sampled frames)` shown; 12-row frame table with timestamps and scores; duration/resolution/FPS/decoded-frames/frames-sampled metadata present; temporal signal and video-level assessment present; `call_10_demo_traffic` present in the simulated report; no AppTest exception or error.
- Cache invalidation: a test uploads the same filename with different bytes and confirms the cached result is invalidated. In browser runs, each new upload produced its own fingerprint and different detection counts; no stale result was reused.
- Corrupt-media check: uploading `broken.jpg` containing `not an image` produced `Could not process 'broken.jpg': cannot identify image file … Please upload a readable JPG/JPEG/PNG image or MP4/MOV/AVI video.` with **0 exceptions** and all six tabs still present.
- Unreadable plate checks: `NOT RUN (NO PLATE REGION)`, `NO OCR RESULTS`, OCR initialization failure, and low-confidence plate cases are covered by AppTest UI tests.

---

## 10. Real functionality vs model limitations

### Real, measured functionality

- File presence, byte counts, decode validity, dimensions, EXIF status, and SHA-256 integrity.
- Faster R-CNN inference, real boxes/labels/confidences, and class counts on three images and sampled video frames.
- Analysis-time orientation recovery without modifying the source file.
- YuNet candidate localisation with a verified model hash, EasyOCR raw text, candidate selection, and strict 0.7 verification gate.
- spaCy/regex extraction from the three exact caller texts, with negation and unknown handling.
- Source-separated fusion with explicit count disagreements and provenance labels.
- Video frame sampling, classifier scores, temporal aggregation, and honest sampling disclosure.
- Six live Streamlit tabs with in-memory uploads, no sample auto-processing, and read-only SQLite.
- Graceful corrupt-media and unreadable-plate handling; SHA-based cache invalidation.

### Model and scope limitations — not hidden

1. **No verified plate on the demo input.** The alphanumeric candidate is 0.4348, far below 0.70. It remains `(unverified)`; no registration number is asserted for dispatch.
2. **YuNet domain mismatch.** The model was trained on Chinese plates; Indian HSRP candidates score low. OCR, not detector score, decides text evidence.
3. **Still-image classifier mismatch.** The frozen ResNet-18 classifier was trained on dashcam video frames. Scores on still images (0.2166, 0.3903, 0.4230) are uncalibrated ranking signals; NO ACCIDENT SIGNAL does not prove normal traffic.
4. **Video is sampled.** Only 12 of 525 frames were analysed, with OCR limited to the top frame. The video-level verdict comes from the highest-scoring sampled frame; temporal evidence is explanatory, not an override.
5. **Caller data is simulated.** Calls are arbitrarily paired with uploads and never become REAL vision evidence. Hospital availability and ambulance positions/live status are not verified; dispatch is explicitly simulated.
6. **Vague quantities stay unknown.** `many` and `several` were not converted into invented numeric counts.

---

## 11. Tests

### Targeted suite

Command:

```bash
.venv/bin/python -m pytest -q \
  tests/test_demo_readiness.py \
  tests/test_demo_input_fixes.py \
  tests/test_detector.py \
  tests/test_plate_detector.py \
  tests/test_plate_reader.py \
  tests/test_caller_extraction.py \
  tests/test_fusion.py \
  tests/test_video_analysis.py \
  tests/test_app_video.py \
  tests/test_app_ui.py
```

Result:

```text
69 passed, 10 warnings in 59.87s
```

`tests/test_demo_input_fixes.py` contains 6 regressions: zero-detection rotation recovery, no unnecessary rotation, padded plate-like OCR fallback, rejection of short non-plate reads, exact mandatory caller descriptions, and forwarding the requested video sample count.

### Full suite

Command:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
218 passed, 11 warnings in 90.22s (0:01:30)
```

Baseline was `204 passed, 6 warnings in 38.38s`. The current suite adds 14 tests and has **0 failures**. Warnings are dependency deprecations/runtime notices from PyTorch, EasyOCR, MPS, and offline Hugging Face caching; none are test failures.

---

## 12. Files changed

| File | Change |
|---|---|
| `core/cv/detector.py` | EXIF transpose and `detect_with_orientation()` quarter-turn recovery with a 0.7 rotation-confidence floor. |
| `core/cv/plate_detector_yunet.py` | Candidate-proposal threshold 0.25; hash-pinned model and documented Chinese-plate limitation retained. |
| `core/cv/plate_detector.py` | Whole-crop EasyOCR fallback, padded candidate regions, strict plate-like filter, and `ocr_confidence` evidence. |
| `core/cv/vehicle_report.py` | Raw and preprocessed plate crops both read; real evidence propagated into vehicle reports. |
| `core/ocr/plate_reader.py` | Raw EasyOCR text/confidence preserved; alphanumeric plate candidates preferred over `IND`. |
| `core/nlp/caller_extraction.py` | `NH`/`SH` route matching, singular person patterns, optional descriptive adjectives, and exact `emergency assistance` phrase. |
| `app/main.py` | Sample-count handling, six-tab disclosures, orientation notice, plate/OCR evidence view, sampling disclaimer, SHA cache key, and graceful error handling. |
| `data/sample_calls/call_09_demo_collision.txt` | Exact collision caller description. |
| `data/sample_calls/call_10_demo_traffic.txt` | Exact traffic caller description. |
| `data/sample_calls/call_11_demo_uncertain.txt` | Exact uncertain caller description. |
| `data/sample_calls/call_demo_01_collision.txt` … `call_demo_04_contradiction.txt` | Additional demo call options; not added to the 8-file benchmark answer key. |
| `tests/test_demo_input_fixes.py` | New targeted regression suite for the measured defects. |
| `tests/test_demo_readiness.py` | Updated demo-input coverage. |
| `DEMO_READINESS_REPORT.md` | Replaced stale report with this measured report. |

No dependencies were installed or upgraded. Original media, `data/sample_calls/expected.json`, and the database contents were not modified. There is no `.git` metadata in this workspace, so changes are uncommitted.

---

## 13. Exact launch command

```bash
cd ~/Projects/third-eye-v2
source .venv/bin/activate
streamlit run app/main.py
```

Open **`http://localhost:8501`** by default.

For this verification run, ports 8501–8503 were already occupied, so the app was started with:

```bash
source .venv/bin/activate
streamlit run app/main.py --server.headless true --server.port 8504
```

and opened at `http://localhost:8504`.

---

## 14. Final decision

The application is ready for a truthful demonstration of all mandatory inputs and all six tabs. It is **PARTIAL**, not a blanket PASS, because the demo plate cannot be verified at the required 0.7 confidence and the video result is explicitly sample-based. Those are real model/scope limitations, correctly surfaced in the UI rather than hidden or fabricated.
