# Phase 8b Notes — Dispatch UI on the Phase 6d Summary

> **Historical notes (superseded 2026-11-25):** These notes reference the former `core/rag/` module paths. The current hospital package is `core/hospitals/`; the UI still preserves the same six tabs and source-labelled display rules. See [SYSTEM_CARD.md](SYSTEM_CARD.md).

## Scope and file ownership

Created/edited only:

- `app/dispatch_ui.py` — new pure helpers: injectable summary loader, value formatting, replay labels, keyless Google Maps URL builder, folium map builder, alternatives rendering, emergency-number helpers.
- `app/main.py` — single-page UI wiring for the new Dispatch tab; shared plain-text emergency-number renderer used by the Hospitals and Dispatch tabs.
- `tests/test_app_ui.py` — widget-policy guard extended; Dispatch "not available yet" test now runs against an injected nonexistent path instead of skipping.
- `tests/test_app_dispatch.py` — new data-backed tests built from the real summary file.
- `docs/phase8b_notes.md` — this file.

No files under `core/`, `db/`, `config/`, `scripts/`, `requirements.txt`, `README.md`, `launch.command`, or any test outside `tests/test_app_*.py` were edited. `folium==1.0.0` and `streamlit-folium==0.28.0` were already installed; nothing was installed. The database was only read through existing read-only helpers; no `save_*` or `init_db` call was added.

## Dispatch data source and honesty rules

The sole dispatch source is `data/outputs/dispatch_summary.json`, written by the concurrent Phase 6d agent. The path is injectable: set `THIRD_EYE_DISPATCH_SUMMARY` to another path, or pass an explicit path to `load_dispatch_summary()` / `configured_summary_path()`. If the file is absent, unreadable, or has no scenarios, the Dispatch tab shows exactly **"Dispatch results not available yet"** and does not crash. Missing or null fields render `not determined`; the explicitly specified OSM fallbacks remain `not listed` (phone), `not specified` (emergency tag), and `not specified in OpenStreetMap` (specialties). Severity is always `not determined (demo placement)`. Bed availability is always `not determined`.

Every displayed value carries a badge: `SIMULATED` for the demo incident, ambulance, ETA, distances, times, replay, ranking heuristic, and alternatives ranking; `REAL` for OSM hospital fields, the summary metadata, disclaimers, and emergency-number records. The static PNG path from `map_png` is resolved relative to the repository root and placed in an expander as an offline fallback.

## Tab descriptions

1. **Scene** — Upload-driven detector boxes on the uploaded image or on the highest-scoring sampled video frame, plus real vehicle/person counts, demo-pairing label, and the visible-exterior limitation.
2. **Vehicles** — In-memory OCR/attribute results for the highest-scoring video frame, plate confidence and `unverified` handling, and the explicit note that OCR is not run on every frame.
3. **Accident check** — Trained-model verdict, uncalibrated score, held-out metrics, per-frame video table/chart, prototype statement, and verbatim model-card limitations.
4. **Incident report** — In-memory fused vision-only or vision-plus-simulated-call report, disagreement flags when a call is selected, and generated incident text.
5. **Hospitals & emergency numbers** — Real OSM facilities ranked from the simulated demo location, with phone, emergency tag, specialties, bed availability, attribution, and limitations. Emergency numbers come from `core/rag/emergency_numbers.py` and are rendered as **plain text** with the line "This app does not place calls; dial from your phone."; `verified` is shown only when the record both says `VERIFIED` and names a source, otherwise `UNVERIFIED`.
6. **Dispatch** — Selectbox of simulated demo incidents from the summary; severity, chosen simulated ambulance and ETA, chosen real OSM hospital (name, phone, emergency tag, specialties, driving vs straight-line distance, transit), total distance/time, the ranking formula labelled "ranking heuristic, not a clinical rating", ambulance/hospital alternatives tables with their reasons and badges, all disclaimers from the file, bed availability `not determined`, an interactive OpenStreetMap folium map with incident/base/hospital markers, two coloured route legs, auto-fit, and a simulated replay-step selectbox that moves a labelled SIMULATED ambulance marker (no timers/buttons/autoplay). The same two plain Google Maps markdown links use the official `https://www.google.com/maps/dir/?api=1` scheme without an API key: base → incident → hospital, and incident → hospital. Their label states that Google's routing and live traffic are not the project's route.

## Interactive map

`build_dispatch_map()` creates `folium.Map` with an explicit `OpenStreetMap` tile layer and attribution, markers for incident (SIMULATED), ambulance base (SIMULATED), hospital (REAL OSM), and the selected replay step (SIMULATED), two `PolyLine` legs in distinct colours, and `fit_bounds()` across base/incident/hospital/leg coordinates. `streamlit_folium.st_folium` renders it; the static PNG sits in an expander beneath it.

## Launch command

```bash
cd ~/Projects/third-eye-v2 && source .venv/bin/activate && streamlit run app/main.py
```

The double-click equivalent is `bash ~/Projects/third-eye-v2/launch.command`.

## Free disk space (raw output)

```text
Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s5   228Gi   159Gi    40Gi    80%    1.5M  423M    0%   /System/Volumes/Data
```

## Test command and raw output

Command: `cd ~/Projects/third-eye-v2 && .venv/bin/python -m pytest tests/test_app_*.py -v`

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /Users/srishanthbiradar/Projects/third-eye-v2/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/srishanthbiradar/Projects/third-eye-v2
configfile: pytest.ini
plugins: anyio-4.16.0
collecting ... collected 13 items

tests/test_app_dispatch.py::test_google_maps_url_builder_uses_official_scheme PASSED [  7%]
tests/test_app_dispatch.py::test_map_builder_returns_folium_map_with_expected_markers PASSED [ 15%]
tests/test_app_dispatch.py::test_replay_step_selection_changes_simulated_marker_position PASSED [ 23%]
tests/test_app_dispatch.py::test_app_scan_finds_no_forbidden_widgets_or_phone_links PASSED [ 30%]
tests/test_app_dispatch.py::test_dispatch_tab_renders_real_summary_fields PASSED [ 38%]
tests/test_app_dispatch.py::test_missing_and_null_fields_render_not_determined PASSED [ 46%]
tests/test_app_dispatch.py::test_emergency_numbers_are_plain_text_and_status_comes_from_data PASSED [ 53%]
tests/test_app_ui.py::test_widget_policy_allows_only_the_cv_uploader PASSED [ 61%]
tests/test_app_ui.py::test_no_upload_runs_no_model PASSED [ 69%]
tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report PASSED [ 76%]
tests/test_app_ui.py::test_dispatch_tab_shows_not_available_with_injected_missing_summary PASSED [ 84%]
tests/test_app_video.py::test_probe_video_reads_real_metadata PASSED     [ 92%]
tests/test_app_video.py::test_frame_sampler_covers_whole_real_clip PASSED [100%]

=============================== warnings summary ===============================
tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/easyocr/detection.py:81: DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10. 
  For migrations of users: 
  1. Eager mode quantization (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode quantize_ API instead 
  2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx,torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization API instead (prepare_pt2e, convert_pt2e)) 
  3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e) 
  see https://github.com/pytorch/ao/issues/2259 for more details
    torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)

tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/ao/quantization/quantize.py:570: DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10. 
  For migrations of users: 
  1. Eager mode quantization (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode quantize_ API instead 
  2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx,torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization API instead (prepare_pt2e, convert_pt2e)) 
  3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e) 
  see https://github.com/pytorch/ao/issues/2259 for more details
    convert(model, mapping, inplace=True)

tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/easyocr/recognition.py:177: DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10. 
  For migrations of users: 
  1. Eager mode quantization (torch.ao.quantization.quantize, torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode quantize_ API instead 
  2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx,torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization API instead (prepare_pt2e, convert_pt2e)) 
  3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e) 
  see https://github.com/pytorch/ao/issues/2259 for more details
    torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)

tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/ao/nn/quantized/dynamic/modules/rnn.py:162: UserWarning: torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will be removed in a future PyTorch release. Please see https://github.com/pytorch/pytorch/issues/184982 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/quantized/Quantizer.cpp:116.)
    w_ih = torch.quantize_per_tensor(

tests/test_app_ui.py::test_real_image_upload_runs_and_shows_vision_only_report
  /Users/srishanthbiradar/Projects/third-eye-v2/.venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py:759: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
    super().__init__(loader)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 13 passed, 6 warnings in 16.23s ========================
```
