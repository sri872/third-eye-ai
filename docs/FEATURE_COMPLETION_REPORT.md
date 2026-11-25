# Feature completion report — PR #1

Date: 2026-11-25
Branch: `remove-rag-retrieval`
PR: https://github.com/sri872/third-eye-ai/pull/1
Scope: Scene, Vehicles/OCR, Accident evidence fusion, four-tier Incident Report, and Dispatch — integrated end to end.

This report records the implemented behaviour and the verification actually run in the PR worktree. It deliberately does not claim calibrated confidence, real dispatch action, live traffic, verified bed availability, or a confirmed collision from a classifier score or box overlap.

## 17 evidence items

### 1. Correct workspace and isolated history

All edits were made only in `/Users/srishanthbiradar/Projects/third-eye-v2-pr1` on `remove-rag-retrieval`. The original workspace (`~/Projects/third-eye-v2`) and the separate `main` worktree were not modified. A recoverable pre-edit backup was created at `/Users/srishanthbiradar/Projects/backups/third-eye-v2-pr1-feature-completion-20261125-230401` before implementation. No history rewrite, force push, or merge was performed.

### 2. Six-tab order preserved

`st.tabs(...)` still creates, in order: `1. Scene`, `2. Vehicles`, `3. Accident check`, `4. Incident report`, `5. Hospitals & emergency numbers`, `6. Dispatch`. AppTest asserts this exact list, and each of the four real demo assets rendered all six tabs during `scripts/verify_streamlit_workflow.py`.

### 3. RAG-free deterministic architecture preserved

No YOLO, SentenceTransformers, FAISS, embedding retrieval, or vector index was added. Hospital suggestions still use the deterministic `core/hospitals/` pipeline: eligibility filtering, OSM source tags, facility category, and Haversine distance. `evaluate_hospital_ranking.py` and `run_hospital_suggestions.py` both report the local OSM cache and the same deterministic ranking basis.

### 4. Real models and real assets retained

The pipeline still uses Faster R-CNN ResNet-50 FPN for detection, the trained frozen ResNet-18 accident classifier, EasyOCR for plate text, YuNet/heuristic plate localisation, cached OSM data, and the real road-graph dispatch demo. No fabricated media, OCR result, hospital, route, or score was introduced.

### 5. One shared structured four-tier contract

`core/schemas.EvidenceAssessment` defines the only assessment contract. `FOUR_TIER_ASSESSMENT_CATEGORIES` is exactly:

1. `ACCIDENT EVIDENCE DETECTED`
2. `POSSIBLE ACCIDENT`
3. `INCONCLUSIVE`
4. `NO CLEAR ACCIDENT EVIDENCE DETECTED`

`core/ml/evidence_assessment.build_accident_assessment()` is the single decision layer. The same dict is attached to `AccidentEvent`, copied into `IncidentReport`, displayed in Accident Check, passed to Incident Report, consumed by Dispatch, persisted in SQLite, and asserted in tests.

### 6. Human review is mandatory and unverifiable categories are not safety claims

`EvidenceAssessment.human_review_required` is validated to always be `True`; a Pydantic test proves it cannot be set to `False`. Accident Check, Tab 4, and Dispatch all render `MANDATORY`. `NO CLEAR ACCIDENT EVIDENCE DETECTED` is explicitly documented as “insufficient positive evidence in this sample”, never “road safe”, “safe”, or “no accident occurred”.

### 7. Raw model verdict remains separate and uncalibrated

`model_verdict` and `model_score` are copied verbatim as diagnostic fields. Tab 3 labels the score “Uncalibrated model score”, Tab 4 has a separate model-prediction section, Dispatch labels both as `MODEL (PREDICTION)`, and the fused audit narrative labels the raw verdict `MODEL (PREDICTION)` rather than `REAL VISION ASSESSMENT`. The assessment category is never recalculated from the score in a UI layer.

### 8. Still-image domain handling is honest

A still photo is outside the dashcam-video training domain. `STILL_IMAGE_SCENE_STATUS = "NOT DETERMINED (out-of-domain still image)"` remains the primary still-image result, while raw model output stays visible separately. The assessment module can emit only `POSSIBLE ACCIDENT` or `INCONCLUSIVE` for still images; it cannot emit machine-confirmed `ACCIDENT EVIDENCE DETECTED` for a single photo.

### 9. No confirmation from overlap, score, or repeated frames

Box overlap, IoU, overlap ratio, person/vehicle intersection, high score, and repeated high sampled-frame scores are recorded as correlated diagnostic cues only. `IMPLEMENTED_IMPACT_METHODS` is intentionally empty because no impact tracker exists, so a fabricated impact sequence cannot confirm an accident. Only an explicit human-verified impact record may produce `ACCIDENT EVIDENCE DETECTED`, and even then human review remains mandatory. Tests cover fake impact methods and sustained high scores without a validated tracker.

### 10. Scene context is pixel-derived and conservative

`analyze_scene_environment()` measures real luminance, contrast, and bottom-region foreground luminance from the supplied pixels; vehicle/person counts come only from detector boxes. Foreground brightness is never labelled asphalt, concrete, shadow, or roadway, and scene type is not inferred when no allowed boxes are detected. Weather is `not determined (requires external atmospheric sensor)` and GPS is `not determined (no GPS sensor in media pixels)`. The Scene tab displays these values plus explicit limitations: no weather, road-surface material, GPS, speed, accident, damage, injury, or road-safety inference.

### 11. Duplicate suppression preserves occlusion and adjacent objects

`suppress_duplicate_detections()` is class-aware and suppresses only near-identical same-class boxes (IoU ≥ 0.85, or containment ≥ 0.95 **with near-equal areas**). A small box inside a much larger box is preserved as a possible occluded object. Both image orientation retries and sampled video frames return `SuppressionResult` diagnostics, so the Scene tab reports real raw versus retained counts for the exact displayed frame. It also states that moderate overlap is preserved and overlap is not collision proof. Existing vehicle-report counts remain unchanged.

### 12. OCR keeps raw text and enforces the 0.70 gate

`read_plate_evidence()` preserves raw OCR text and raw confidence verbatim. `is_verified_plate_confidence()` uses the one fixed `0.70` threshold shared by the OCR and UI modules. Candidate ranking prefers registration format, then alphanumeric text, then partial text, with real confidence breaking ties. Verification requires both `format_valid` and the `0.70` gate: format-valid text is a ranking hint only, and a high-confidence partial read stays `NOT READABLE (UNVERIFIED)` / `PLATE IDENTIFICATION: UNVERIFIED` with raw text retained. Deterministic preprocessing variants are cached by pixel hash, and the app calls `read_plate_evidence()` once per associated plate region rather than ranking crops itself.

### 13. Vehicle-to-plate association is geometry-based and explicit

`associate_plate_regions()` uses containment and centre geometry only. It returns `ASSOCIATED`, `AMBIGUOUS`, `UNASSOCIATED`, or `NO PLATE REGION DETECTED`; two plausible regions overlapping the same vehicle produce `AMBIGUOUS` and no auto-selection. The upload UI now calls this helper before OCR instead of taking `regions[0]`, and `vehicle_report.py` carries the same status into `plate_detection_status` without inventing confidence. UI regressions prove ambiguous regions skip OCR entirely.

### 14. Incident Report carries the exact assessment with source separation

`fuse_incident_evidence()` copies `AccidentEvent.evidence_assessment` into `IncidentReport.evidence_assessment` and writes the exact category into the narrative. The no-call vision-only audit narrative now includes the same shared assessment block rather than claiming “normal traffic flow”; a low classifier score is recorded as insufficient positive evidence, never road safety or no accident. Tab 4 section `8. Four-tier accident evidence assessment` displays category, raw verdict, uncalibrated score, media type, domain compatibility, scene status, evidence, contradictions, uncertainty, mandatory review, reasoning, temporal availability, and data source without recalculating the tier. Sections 1–7 and the final all-fields inventory remain stable; cells are truncated only for display and never exceed 120 characters.

### 15. SQLite persistence is additive and round-trips the contract

`accident_events` and `incident_reports` gained nullable `evidence_assessment TEXT` columns in DDL plus backward-compatible `ALTER TABLE` migrations. `save_accident_event()` and `save_incident_report()` serialise the exact assessment as sorted JSON. A new database regression test saves and reads both tables and asserts `json.loads(...) == assessment.as_dict()`.

### 16. Dispatch consumes the same assessment as context only

`assessment_dispatch_rows()` and `assessment_dispatch_notice()` display the exact supplied category, mandatory review, raw verdict/score, media/scene context, evidence summary, and an explicit rule: the assessment is context only and does not dispatch, clear, or prioritize an incident; dispatcher verification is required. Human review is labelled `OPERATIONAL RULE`, and the four-category tuple is imported from `core.schemas` instead of duplicated. Ranking, routing, ETA, replay, and emergency-number functions were not changed. Official numbers remain plain-text emergency contacts separate from simulated routing actions. Tests assert all four categories pass through, missing assessments show `not determined`, and no forbidden safety claim appears.

### 17. Verification actually executed

| Check | Result |
|---|---|
| Full pytest (`../third-eye-v2/.venv/bin/python -m pytest -q`) | **327 passed, 14 warnings in 61.70s** |
| New regression tests | `test_evidence_assessment.py`, `test_scene_tab.py`, `test_plate_association.py`, `test_incident_report_improvements.py`, plus expanded AppTest/OCR association, narrative, dispatch, and reader tests; all pass |
| `git diff --check` | Clean, no whitespace errors |
| `compileall -q app core db scripts tests` | Clean |
| `scripts/verify_demo_assets.py` | Real inference on all demo assets; regenerated `data/outputs/demo_verification.json` during verification |
| `scripts/verify_streamlit_workflow.py` | Four real uploads, six tabs each, SHA-256/cache invalidation, source separation, OSM text, simulated dispatch, sampled-frame rows; regenerated `data/outputs/streamlit_workflow.json` during verification |
| `scripts/run_detection.py`, `run_vehicle_reports.py`, `run_accident_analysis.py`, `run_incident_reports.py`, `run_hospital_suggestions.py`, `evaluate_hospital_ranking.py`, `run_dispatch.py` | All completed with exit code 0; 45 report pairings, 115 OSM facilities/102 dispatch-eligible, four simulated route scenarios with no live traffic |
| `scripts/evaluate_accident_model.py` | **Blocked honestly**: `FileNotFoundError: data/embeddings.npz` (exit 1). No metrics were fabricated and no dependency install was attempted. |

Representative real-asset observations: `single_car_plate.jpg` produced raw OCR text `IND` (raw confidence `0.9457`) and a format-valid candidate `TS07JA0566` at `0.4399`, correctly left **unverified**; `two_car_collision.jpg` scored `0.2166`; `heavy_traffic.jpg` scored `0.3903`; `road vdo.mp4` had 525 frames at 25 fps/21 s, 12 evenly sampled frames, temporal signal `ISOLATED UNCERTAIN FRAME IN SAMPLE`, peak score `0.4201`, and no accident confirmation.

The verification scripts rewrote tracked `data/outputs/*` artifacts (including two route map PNGs). Those derived artifacts were restored after verification so this change set contains only implementation, tests, and this audit report; the recorded observations above are from the real runs, not from regenerated files in the final diff.

## Conclusion

The five workstreams are integrated rather than side-by-side: the same `EvidenceAssessment` object now flows from analysis into Accident Check, Incident Report, Dispatch, SQLite, and tests. All available verification succeeded; the only non-passing verification script is the pre-existing, data-blocked accident-model evaluator that requires `data/embeddings.npz`, which is absent in this repository.


---

## Addendum — separate emergency-knowledge RAG assistant restored (2026-11-26)

The embedding-based **hospital** retrieval system remains removed. Hospital discovery and eligibility still use the deterministic `core/hospitals/` package.

A separate RAG knowledge assistant has now been restored under `core/rag/` for emergency-response knowledge questions in Tab 5:

- `core/rag/retriever.py`: local scikit-learn TF-IDF lexical passage retrieval.
- `core/rag/generator.py`: optional local Ollama generation with a strict source-passage prompt; labelled extractive response fallback if Ollama is not configured or unavailable.
- `core/rag/pipeline.py`: end-to-end retrieval, response, source links, and limitations.
- `data/knowledge_base/emergency_response_india.json`: curated and paraphrased source passages from MoRTH, the Motor Vehicles (Amendment) Act, 2019, NHM, and WHO official publications.
- Tab 5 adds a dropdown of preset knowledge questions, retrieved evidence, relevance scores, and official source links.

This RAG assistant is separate from hospital ranking, accident evidence assessment, incident report, and dispatch. It does not influence accident verdicts, eligibility, bed availability, triage, or emergency dispatch.

The default mode is **extractive RAG**: retrieved passages are displayed as an extractive evidence summary. Actual LLM-generated text is optional and requires a separately installed and running local Ollama model configured through `THIRD_EYE_RAG_OLLAMA_MODEL`. The project does not install or download an LLM.

New tests were added in `tests/test_rag_knowledge.py`, and `tests/test_hospitals_architecture.py` now verifies both that the separate knowledge RAG package exists and that deterministic hospital ranking does not depend on it.

**Validation note:** the 327-test / 14-warning result above is from the integration before this RAG addendum. The newly added RAG and UI tests have not yet been run in the Mac PR worktree; rerun the focused tests and full pytest suite before merging this addendum.
