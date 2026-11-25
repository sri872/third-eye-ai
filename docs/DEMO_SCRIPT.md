# Phase 10 Live Demo Script (~5 minutes)

Run from `/Users/srishanthbiradar/Projects/third-eye-v2` with `.venv` active. Start the dashboard with `bash launch.command`, open `http://localhost:8501`, and use the sidebar uploader (one image or one video at a time). Nothing is written to the project or database by an upload.

The expected outputs below are **verbatim real output** from running the pipeline functions on this machine: `app.video_utils.sample_video_frames` (the same 12-frame sampler the dashboard uses), `core.ml.accident_analyzer.classify_frame_probability` / `determine_verdict`, `core.cv.detector.detect`, `core.cv.vehicle_attributes.crop_vehicle` / `dominant_color` / `get_vehicle_type`, and `core.ocr.plate_reader.read_plate`. `app/` was imported read-only; no file under `app/` was edited.

## Step 1 — Set expectations (30 s)

Say: "This is a local prototype. Detector boxes, OCR text/confidence, video metadata and model scores come from the uploaded media (REAL). The selected incident location, the 6 ambulances and their ETAs are SIMULATED. Road distances come from a cached OpenStreetMap graph with no live traffic. Google Maps links in the Dispatch tab are external."

## Step 2 — Show the held-out split provenance (30 s)

Command actually run:

```bash
.venv/bin/python - <<'PY'
import json
m = json.load(open('models/accident_classifier_metrics.json'))
print('accuracy', m['test_accuracy'], 'roc_auc', m['test_roc_auc'], 'test_videos', m['splits']['test_videos'])
print('75 in test ids:', 75 in m['test_video_ids'], '1023 in test ids:', 1023 in m['test_video_ids'])
for frame in m['test_frames']:
    if frame['video_id'] in ('75', '1023'):
        print(frame)
PY
```

Raw output:

```text
accuracy 0.6778 roc_auc 0.7452 test_videos 60
75 in test ids: True 1023 in test ids: True
{'frame_id': '00075_pos60_173', 'video_id': '75', 'label': 1, 'prob': 0.5965451002120972}
{'frame_id': '00075_pos75_216', 'video_id': '75', 'label': 1, 'prob': 0.6754897236824036}
{'frame_id': '00075_pos90_259', 'video_id': '75', 'label': 1, 'prob': 0.8242461085319519}
{'frame_id': '01023_pos60_170', 'video_id': '1023', 'label': 1, 'prob': 0.196553573012352}
{'frame_id': '01023_pos75_212', 'video_id': '1023', 'label': 1, 'prob': 0.252641499042511}
{'frame_id': '01023_pos90_255', 'video_id': '1023', 'label': 1, 'prob': 0.31014132499694824}
```

Both selected clips are positive-labelled members of the held-out test split. Video 75 is scored above 0.5 at all three metric sample positions; video 1023 is scored below 0.5 at all three — that is the honest failure.

## Step 3 — Upload the demo image (60 s)

Upload `data/samples/road_1.jpg`. Point to the Scene tab (boxes/counts), Vehicles tab (colour/plate) and Accident check tab (single-frame score). Expected real output from the pipeline run:

```text
/Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/ao/nn/quantized/dynamic/modules/rnn.py:162: UserWarning: torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will be removed in a future PyTorch release. Please see https://github.com/pytorch/pytorch/issues/184982 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/quantized/Quantizer.cpp:116.)
  w_ih = torch.quantize_per_tensor(
/Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py:759: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  super().__init__(loader)
=== DEMO IMAGE ===
path: /Users/srishanthbiradar/Projects/third-eye-v2/data/samples/road_1.jpg
size: (1771, 1237)
[Third-Eye CV] Initializing Faster R-CNN ResNet-50 FPN on device: MPS
detections: 8 vehicles: 6 persons: 2
[Third-Eye Accident Analyzer] Loading frozen ResNet-18 on device: MPS
classifier_score: 0.31794 verdict: ('NO ACCIDENT SIGNAL', 'none', 'normal_flow')
[Third-Eye OCR] Initializing EasyOCR Reader on device: CPU
vehicle 1: type=car color=red share=0.4439 det_conf=0.999 plate=CAN09G plate_conf=0.5412
vehicle 2: type=car color=green share=0.2985 det_conf=0.9988 plate=None plate_conf=None
vehicle 3: type=car color=red share=0.5779 det_conf=0.9912 plate=None plate_conf=None
vehicle 4: type=car color=grey share=0.7336 det_conf=0.9155 plate=None plate_conf=None
vehicle 5: type=car color=grey share=0.5938 det_conf=0.6039 plate=None plate_conf=None
vehicle 6: type=car color=green share=0.4912 det_conf=0.5256 plate=None plate_conf=None
```

Talking points:
- **Known out-of-domain miss**: `road_1.jpg` shows a real two-car collision, but the accident classifier scores it **0.3179** (`NO ACCIDENT SIGNAL`). Explain this honestly to the audience: the model was trained exclusively on forward-facing Nexar dashcam video clips; a single still photo from an oblique roadside perspective represents an out-of-domain distribution shift. The geometric overlap analysis detects `max vehicle IoU=0.12, heavily overlapping pairs=7`, but by design geometric heuristics serve only as explanatory context and never override the model verdict.
- **Plate reader diagnostic limit**: only 1 of 6 vehicles produced plate text, with real OCR confidence 0.5412 — below the dashboard's 0.7 `unverified` threshold. Do not call this a reliable plate reader.

## Step 4 — Upload the held-out accident clip the model gets right (90 s)

Upload `/Users/srishanthbiradar/Datasets/ThirdEye/Nexar/test-public/positive/00075.mp4`. In the Accident check tab the table/chart should show 8 of 12 sampled frames at or above 0.60. Verbatim function output:

```text
=== DEMO VIDEO ===
path: /Users/srishanthbiradar/Datasets/ThirdEye/Nexar/test-public/positive/00075.mp4
metadata: frames=289 fps=30.0 width=1280 height=720 duration_s=9.633
frame= 1 source_frame=   0 t=  0.000s score=0.539433 verdict=UNCERTAIN vehicles=6 persons=0
frame= 2 source_frame=  26 t=  0.867s score=0.649817 verdict=LIKELY ACCIDENT vehicles=4 persons=0
frame= 3 source_frame=  52 t=  1.733s score=0.835425 verdict=LIKELY ACCIDENT vehicles=3 persons=0
frame= 4 source_frame=  79 t=  2.633s score=0.669323 verdict=LIKELY ACCIDENT vehicles=1 persons=0
frame= 5 source_frame= 105 t=  3.500s score=0.449300 verdict=UNCERTAIN vehicles=3 persons=0
frame= 6 source_frame= 131 t=  4.367s score=0.577594 verdict=UNCERTAIN vehicles=5 persons=0
frame= 7 source_frame= 157 t=  5.233s score=0.410082 verdict=UNCERTAIN vehicles=6 persons=0
frame= 8 source_frame= 183 t=  6.100s score=0.724450 verdict=LIKELY ACCIDENT vehicles=11 persons=1
frame= 9 source_frame= 209 t=  6.967s score=0.782801 verdict=LIKELY ACCIDENT vehicles=5 persons=0
frame=10 source_frame= 236 t=  7.867s score=0.737955 verdict=LIKELY ACCIDENT vehicles=4 persons=3
frame=11 source_frame= 262 t=  8.733s score=0.868467 verdict=LIKELY ACCIDENT vehicles=10 persons=0
frame=12 source_frame= 288 t=  9.600s score=0.875386 verdict=LIKELY ACCIDENT vehicles=11 persons=0
top_frame=12 source_frame=288 t=9.6s top_score=0.875386 frames>=0.60=8
```

## Step 5 — Upload the held-out accident clip the model gets wrong (90 s)

Upload `/Users/srishanthbiradar/Datasets/ThirdEye/Nexar/test-public/positive/01023.mp4`. This clip is labelled positive in the held-out split, but the model never reaches 0.60 on any sampled frame. Show this before moving on — it is the honest failure case. Verbatim function output:

```text
=== DEMO VIDEO ===
path: /Users/srishanthbiradar/Datasets/ThirdEye/Nexar/test-public/positive/01023.mp4
metadata: frames=284 fps=30.0 width=1280 height=720 duration_s=9.467
frame= 1 source_frame=   0 t=  0.000s score=0.207900 verdict=NO ACCIDENT SIGNAL vehicles=17 persons=0
frame= 2 source_frame=  26 t=  0.867s score=0.248185 verdict=NO ACCIDENT SIGNAL vehicles=20 persons=0
frame= 3 source_frame=  51 t=  1.700s score=0.139082 verdict=NO ACCIDENT SIGNAL vehicles=18 persons=0
frame= 4 source_frame=  77 t=  2.567s score=0.116356 verdict=NO ACCIDENT SIGNAL vehicles=21 persons=0
frame= 5 source_frame= 103 t=  3.433s score=0.197019 verdict=NO ACCIDENT SIGNAL vehicles=18 persons=0
frame= 6 source_frame= 129 t=  4.300s score=0.151156 verdict=NO ACCIDENT SIGNAL vehicles=20 persons=0
frame= 7 source_frame= 154 t=  5.133s score=0.231850 verdict=NO ACCIDENT SIGNAL vehicles=24 persons=0
frame= 8 source_frame= 180 t=  6.000s score=0.271732 verdict=NO ACCIDENT SIGNAL vehicles=19 persons=0
frame= 9 source_frame= 206 t=  6.867s score=0.341071 verdict=NO ACCIDENT SIGNAL vehicles=22 persons=0
frame=10 source_frame= 232 t=  7.733s score=0.418328 verdict=UNCERTAIN vehicles=22 persons=0
frame=11 source_frame= 257 t=  8.567s score=0.445759 verdict=UNCERTAIN vehicles=29 persons=0
frame=12 source_frame= 283 t=  9.433s score=0.502151 verdict=UNCERTAIN vehicles=24 persons=0
top_frame=12 source_frame=283 t=9.433s top_score=0.502151 frames>=0.60=0
```

## Step 6 — Dispatch and results (60 s)

Open the Dispatch tab. Pick any of the four simulated incidents. Say: "Hospital coordinates and names are REAL OpenStreetMap data; the ambulance positions, ETAs and replay marker are SIMULATED; there is no live traffic. The ranking formula is a heuristic, not a clinical rating." The source file is `data/outputs/dispatch_summary.json`; maps and GeoJSON are in `data/outputs/routes/`. Google Maps links are external and are not the project's route.

## Likely viva questions (with honest answers)

1. **Why not YOLO?** The project policy in `README.md` explicitly prohibits YOLO/Ultralytics. Detection uses COCO-pretrained Faster R-CNN ResNet-50 FPN from TorchVision; no YOLO model was trained or measured here, so no YOLO accuracy claim is made.
2. **Why simulated ambulances?** There is no live GPS/fleet feed in this workspace. `config/simulated_ambulances.json` places 6 units at real OSM hospital coordinates, and every dispatch record is tagged `data_source: SIMULATED`. The ETAs are computed on the cached OSM road graph, not from actual vehicles.
3. **Why only 68% accuracy?** `models/accident_classifier_metrics.json` records 67.78% accuracy and ROC-AUC 0.7452 on 180 frames from 60 held-out forward-facing dashcam videos (confusion matrix TN=56, FP=34, FN=24, TP=66). The sample is small, dashcam-specific, and trained on imminent-impact frames; scores are uncalibrated, and the failure clip above shows the miss directly.
4. **What does OpenStreetMap lack here?** Per `docs/phase5_report.md`: of 116 facilities, only 3 (2.59%) have an `emergency=yes/no` tag, 2 (1.72%) have `healthcare:speciality`, and 17 (14.66%) have a phone. The chosen demo hospitals show `emergency_tag: null` and no specialty tags in the dispatch JSON, so suitability is inferred from names by a rule, not from clinical records.
5. **How is this different from the earlier submitted THIRD EYE AI report?** This version has no FastAPI backend, no login/tenants, no SHA-256 evidence custody, no video tracking and no audio transcription. It adds a trained accident classifier with held-out metrics, real-road routing with simulated ambulances, a manual-upload Streamlit dashboard, SQLite persistence, and honest REAL/SIMULATED labelling.
6. **Why does the system say no accident on an obvious collision photo?** (e.g. `data/samples/road_1.jpg` or `road_2.jpg`). The accident classifier was trained strictly on forward-facing dashcam video frames from the Nexar dataset. Single still photographs taken from roadside, overhead, or oblique perspectives are out of domain for the feature extractor and classifier. On `road_1.jpg`, the model outputs score 0.3179 (`NO ACCIDENT SIGNAL`). While the geometric overlap heuristic notes 7 heavily overlapping pairs (max vehicle IoU 0.12), the architecture intentionally prevents heuristics from overriding the trained model's decision, making this an honest and documented domain-shift false negative.
