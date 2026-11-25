# Phase 14 Report — Video Verification

Date: 2026-11-25  
Workspace: `/Users/srishanthbiradar/Projects/third-eye-v2`  
Python: `3.12.3`  
No dependencies installed. No full-clip inference was added. No accident was confirmed from sampled frames.

## 1. Scope preserved

- All six Streamlit tabs remain intact.
- The trained classifier verdict remains authoritative.
- Geometric rules remain explanatory and out of runtime use.
- Still images still show `NOT DETERMINED (out-of-domain still image)` in the scene assessment context.
- Plate verification remains at `PLATE_TRUST_THRESHOLD = 0.7`.
- REAL/SIMULATED evidence stays separated.
- No live dispatch, live traffic, confirmed accident, or verified plate is claimed.

## 2. Configurable frame sampling

`app/main.py` now exposes a sidebar control:

- `Video frames to sample`: `4–60`, default `12`.
- The configured count is part of the session cache key, so changing it reruns the video analysis.
- `_process_video()` accepts `sample_count` and clamps it to `[2, MAX_SAMPLE_COUNT=60]`.
- `app/video_utils.sample_video_frames()` already supported an explicit `sample_count`; Phase 14 tests now verify it honours a non-default value.

Long-video messaging is dynamic: instead of hard-coding “12 frames”, the UI reports the actual number of sampled frames.

## 3. Frame-level evidence

Each sampled frame row now contains:

- chronological sample position
- real source frame index
- timestamp derived from container FPS
- uncalibrated classifier score
- verdict/severity from existing deployment bands
- vehicle count and person count from the real detector

The same rows are stored in `result["frame_rows"]` and inside `result["video_evidence"]["frame_rows"]`.

## 4. Temporal analysis

New module: `app/video_analysis.py`

Functions:

- `analyze_frame_scores(rows)` computes min/max/mean score, band counts, longest consecutive run of `>=0.60`, longest run of `>=0.40`, and a descriptive temporal signal.
- `build_video_assessment(metadata, rows, temporal, verdict)` combines real container metadata with the frame-level evidence while explicitly stating that temporal evidence does not override the trained model.

Temporal labels are descriptive only:

- `SUSTAINED HIGH SIGNAL IN SAMPLED FRAMES`
- `ISOLATED HIGH FRAME IN SAMPLE`
- `SUSTAINED UNCERTAIN SIGNAL IN SAMPLED FRAMES`
- `ISOLATED UNCERTAIN FRAME IN SAMPLE`
- `NO HIGH OR UNCERTAIN FRAME IN SAMPLE`

No label is an accident verdict. The `video_assessment` text explicitly says “not an accident confirmation”.

## 5. Result and report propagation

`_process_video()` result now includes:

- `temporal_evidence`
- `video_evidence`
- `video_assessment`
- `video_temporal_signal`

Tab 3 now shows:

- sampling method and configured limit
- per-frame table and score chart
- temporal signal, score range, band counts, consecutive-run counts
- video-level assessment
- authoritative model verdict from the highest-scoring sampled frame

Fusion (`core/nlp/fusion.py`) now accepts `video_evidence` and adds a clearly tagged section:

`5. REAL VIDEO FRAME/TEMPORAL EVIDENCE (Source: REAL; sampled frames only)`

`IncidentReport` gains optional fields:

- `video_evidence`
- `video_assessment`
- `video_temporal_signal`

SQLite `incident_reports` gains backward-compatible columns for the same fields; legacy tables are migrated by `init_db()`.

The vision-only in-memory report also appends the video assessment text and carries the new fields. The paired-call path passes `result["video_evidence"]` into fusion.

## 6. Real video evidence

Diagnostic script:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase14_real_video_diag.py`  
Output:
`/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase14_real_video_diag.out`

Video:

```text
/Users/srishanthbiradar/Desktop/road vdo.mp4
bytes:    14323936
SHA-256:  cdc01a90b62616b62abeeaa8110de9314a60c4dce0d140820cbd7d54c7a76809
metadata: 525 frames, 25.0 fps, 1920x1080, 21.0 s
```

### Default configuration: 12 sampled frames

- Highest score: `0.4201` at source frame `381`, timestamp `15.24 s`, verdict `UNCERTAIN`.
- Score range: `0.1915–0.4201`, mean `0.2872`.
- Bands: `0` frames `>=0.60`, `1` frame in `0.40–0.60`, `11` frames `<0.40`.
- Longest `>=0.60` run: `0`; longest `>=0.40` run: `1`.
- Temporal signal: `ISOLATED UNCERTAIN FRAME IN SAMPLE`.
- Assessment remains `UNCERTAIN`; no accident confirmation.

### Increased configuration: 24 sampled frames

- Highest score: `0.4594` at source frame `410`, timestamp `16.4 s`, verdict `UNCERTAIN`.
- Score range: `0.1128–0.4594`, mean `0.2854`.
- Bands: `0` frames `>=0.60`, `3` frames in `0.40–0.60`, `21` frames `<0.40`.
- Longest `>=0.60` run: `0`; longest `>=0.40` run: `3`.
- Temporal signal: `SUSTAINED UNCERTAIN SIGNAL IN SAMPLED FRAMES`.
- Assessment remains `UNCERTAIN`; no accident confirmation.

The change in sample count changed the observed peak and temporal label. This is expected for an even-sample review and is reported honestly; it is not treated as ground truth.

## 7. Test evidence

Targeted Phase 14 suite:

```text
.venv/bin/pytest -q tests/test_video_analysis.py tests/test_app_video.py \
  tests/test_fusion.py tests/test_database.py tests/test_app_ui.py
32 passed, 5 warnings in 26.80s
```

Saved output: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase14_targeted_1.txt`.

Full suite after Phase 14:

```text
.venv/bin/pytest --cache-clear -q
200 passed, 6 warnings in 36.25s
```

Saved output: `/private/var/folders/bn/2bj93s1s2zz_lshjg99fp0200000gn/T/phase14_full_suite.txt`.

## 8. Honest limitations

- Sampling is still even-spaced; it is not full-clip inference, event-triggered sampling, or continuous temporal tracking.
- Temporal labels are computed from uncalibrated scores and sampled frames only.
- No labelled real accident video ground truth exists; video-level accuracy remains unverifiable.
- A sustained uncertain run is not a confirmed accident.
- Increasing sample count changes which frames are seen; results are configuration-dependent.
- No OCR is run on every frame.
- Git diff/status remain unavailable because the workspace has no `.git` directory.

## 9. Phase 14 outcome

Phase 14 is complete. Video sampling is configurable, frame-level evidence is explicit, temporal patterns are computed from real classifier outputs, video-level assessments propagate through UI, fusion, report schemas, and SQLite, and real-video diagnostics demonstrate both the 12-frame default and a 24-frame alternative without making an accident claim.
