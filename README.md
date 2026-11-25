# Third Eye AI 🚑

**Local multimodal accident-analysis and simulated ambulance-dispatch prototype**

Third Eye AI analyses an uploaded road image or traffic video, detects vehicles and people, attempts number-plate OCR, runs a trained accident classifier, extracts structured facts from an optional caller transcript, ranks source-labelled hospital records deterministically, and displays a simulated dispatch plan. It is designed to run locally on a MacBook Air M2 using local model weights and cached OpenStreetMap data.

> **Prototype, not an emergency service.** The app does not place calls, contact ambulances, verify live GPS or traffic, confirm hospital bed availability, or establish an accident from a single image. Machine outputs are evidence for human review, not emergency decisions. Values are labelled by provenance such as REAL, MODEL PREDICTION, SIMULATED, or NOT DETERMINED.

---

## Core AI technologies and supporting geospatial intelligence

| # | Technology | Implementation | What executes |
|---|---|---|---|
| 1 | Computer Vision | `core/cv/` | TorchVision Faster R-CNN ResNet-50 FPN detects supported COCO classes; OpenCV handles image processing, plate-region preparation, annotations, and detection diagnostics. |
| 2 | OCR | `core/ocr/plate_reader.py` | Local EasyOCR reads candidate plate crops. Raw OCR text and confidence are retained. A candidate is not treated as verified merely because it matches an Indian registration format; the configured confidence gate is 0.70. |
| 3 | Machine Learning | `core/ml/accident_analyzer.py`, `core/ml/evidence_assessment.py` | A trained classifier uses frozen ResNet-18 visual features and a logistic-regression model. Its raw verdict and score remain separate diagnostic fields; the score is uncalibrated and is not itself an accident confirmation. |
| 4 | Deep Learning | `core/cv/detector.py`, `core/ml/accident_analyzer.py` | Neural-network inference uses Faster R-CNN for object detection and ResNet-18 feature extraction for the accident classifier. |
| 5 | NLP | `core/nlp/` | spaCy/rule-based extraction structures optional caller-reported vehicles, people, location, injuries, hazards, urgency, and accident status. Caller information remains labelled separately from real visual evidence. |
| 6 | Geospatial intelligence (supporting module) | `core/hospitals/`, `core/routing/` | Hospital records are classified and ranked using source tags, facility category, and geographic distance. Routing is a clearly labelled simulation using the available road-network data. |

### RAG (Retrieval-Augmented Generation): restored as a separate knowledge assistant

**RAG is implemented again as a separate, source-grounded knowledge feature in Tab 5.** It retrieves relevant passages from the bundled emergency-response knowledge base and presents answers with source links.

The restored implementation lives in `core/rag/`:

1. `core/rag/retriever.py` builds a local TF-IDF retrieval index with scikit-learn and retrieves relevant passages from the curated corpus.
2. `core/rag/generator.py` uses an optional local Ollama model to write a grounded answer with passage citations.
3. When Ollama is not configured or is unavailable, the assistant clearly switches to an extractive evidence summary. The UI labels the mode so it never pretends that an LLM generated the response.
4. `core/rag/pipeline.py` returns the answer, retrieved passages, source metadata, retrieval mode, and safety limitations.
5. `data/knowledge_base/emergency_response_india.json` contains curated, paraphrased source summaries from official Government of India and WHO publications.

The current RAG corpus covers Good Samaritan rights and protections, Section 134A of the Motor Vehicles (Amendment) Act, 2019, NHM 108/102 ambulance-service descriptions, and WHO road-safety policy guidance. Every document record stores its source agency and official URL.

**Important separation:** the restored RAG assistant is for emergency-response knowledge questions only. Hospital discovery and emergency eligibility continue to use deterministic structured-data ranking in `core/hospitals/`:

1. Source-reported emergency eligibility.
2. Source-derived facility category.
3. Straight-line geographic proximity (Haversine distance) as a secondary ordering signal.

RAG does not decide whether an accident occurred, make a clinical triage decision, determine hospital eligibility, verify beds or ambulance availability, or issue dispatch instructions. This separation keeps hospital ranking explainable while restoring a genuine retrieval-augmented knowledge capability.

#### Local generation (optional)

The assistant works without a remote API. By default it uses a labelled extractive fallback. To enable local LLM generation, run Ollama separately, pull a compatible model, and set:

```bash
export THIRD_EYE_RAG_OLLAMA_MODEL="qwen2.5:3b"
export THIRD_EYE_RAG_OLLAMA_HOST="http://127.0.0.1:11434"
```

No Ollama package or model is installed or downloaded by the project. See [RAG_KNOWLEDGE_ASSISTANT.md](docs/RAG_KNOWLEDGE_ASSISTANT.md) for architecture, source scope, configuration, and safety limits.

**NLP is not the same as RAG.** The existing NLP module extracts structured details from caller text. The restored RAG module separately retrieves relevant document passages and grounds a knowledge response in those passages.

---

## Six Streamlit tabs

The entry point is `app/main.py`. The six tabs remain in this order:

1. **Scene** — real detector boxes, supported object counts, duplicate-suppression diagnostics, and conservative pixel-derived scene context. The app does not infer road material, weather, speed, damage, or injuries from brightness or contrast.
2. **Vehicles** — vehicle crops and attributes, plate-region localisation, raw OCR text/confidence, candidate formatting, and geometry-based association states such as ASSOCIATED, AMBIGUOUS, or UNASSOCIATED. OCR candidates remain unverified without the required confidence and format checks.
3. **Accident Check** — the shared four-tier `EvidenceAssessment`: `ACCIDENT EVIDENCE DETECTED`, `POSSIBLE ACCIDENT`, `INCONCLUSIVE`, or `NO CLEAR ACCIDENT EVIDENCE DETECTED`. The original model verdict and uncalibrated score are shown separately. A still image always has the primary scene status `NOT DETERMINED (out-of-domain still image)`. Overlapping 2-D boxes, a high classifier score, or person/vehicle proximity alone do not prove a collision.
4. **Incident Report** — structured, source-labelled report sections using the exact assessment produced by the evidence pipeline. The report does not independently recalculate the assessment. REAL vision, MODEL PREDICTION, OCR CANDIDATE, SIMULATED CALLER, COMPUTED COMPARISON, and NOT DETERMINED values remain distinguishable.
5. **Hospitals & Emergency Numbers** — source-labelled OpenStreetMap hospital/facility suggestions ranked deterministically, with ranking explanations and plain-text emergency contact information. It also contains the RAG emergency-knowledge assistant with source passages and citations. The app does not place calls or verify live bed availability.
6. **Dispatch** — uses the same assessment as context and shows the supported simulated hospital/route workflow. It does not dispatch an actual ambulance, make an emergency decision, or claim live GPS or traffic.

Human review is mandatory for every evidence assessment. The four-tier category is not a medical diagnosis or an official accident determination.

---

## Architecture and evidence flow

```text
Image or video upload
        |
        +--> Computer Vision
        |      Faster R-CNN + OpenCV
        |      -> detections, boxes, counts, suppression diagnostics
        |
        +--> Vehicles and OCR
        |      plate-region localisation + EasyOCR
        |      -> raw text, confidence, candidate, association status
        |
        +--> Accident analysis
        |      classifier diagnostics + available visual/context evidence
        |      -> one shared EvidenceAssessment contract
        |
        +--> Incident Report
        |      exact same EvidenceAssessment + source-labelled report fields
        |
        +--> Hospitals
        |      structured OpenStreetMap records
        |      -> deterministic eligibility/category/distance ranking
        |
        +--> Dispatch
               same assessment as context + labelled simulated route
```

The shared assessment is passed through Accident Check, `AccidentEvent`, `IncidentReport`, Dispatch context, SQLite persistence, and audit output. Downstream modules consume that assessment instead of independently generating conflicting verdicts.

For video, the application uses the actual supported sampling and frame-analysis path. Sampled-frame results must not be described as full-frame, continuous collision tracking unless that capability is explicitly implemented and verified.

---

## Installation

Python 3.12 and macOS are the tested environment. From a clone of this repository:

```bash
cd third-eye-v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hospital search uses deterministic structured-data rules and does not use RAG or semantic similarity. The separate knowledge assistant uses scikit-learn TF-IDF retrieval (already included in requirements) and has an extractive fallback. Optional LLM generation requires a separately installed local Ollama service; no paid API key is required.

### Model and data assets

The repository includes trained and OCR assets where licensing and repository size permit. TorchVision model weights may be downloaded into the local PyTorch cache on first use. Preserve the existing model files and sample data when updating the project.

---

## Launch

```bash
cd third-eye-v2
source .venv/bin/activate
streamlit run app/main.py
```

Open <http://localhost:8501>.

---

## Demonstration sequence

Use the real project samples in `data/samples/demo/` where available. Upload media through the application; do not rely on screenshots of the interface as model inputs.

1. **Single-car/plate sample** — demonstrate Scene detections, plate-region localisation, raw OCR text, confidence, and the unverified-candidate state.
2. **Two-car collision sample** — demonstrate visible scene evidence, conservative four-tier assessment, mandatory human review, and the matching Incident Report assessment. Because this is a still image, the primary scene status remains NOT DETERMINED (out-of-domain still image).
3. **Heavy-traffic sample** — demonstrate multiple-object detection and that crowded or closely spaced vehicles do not automatically become confirmed accident evidence.
4. **Road video sample** — demonstrate the actual sampled-frame video workflow and its sampling limitations.
5. **Hospitals and Dispatch** — show the deterministic ranking explanation, source attribution, emergency contact information, and simulated route disclaimers. In the RAG knowledge assistant dropdown, select a question about Good Samaritan protections or 108/102 services and inspect the retrieved passages and official source links.

Generated or synthetic images are useful for illustrative software demonstrations, but they are not ground-truth accident evidence. OCR output from a synthetic or ordinary image remains an unverified candidate unless an independent verification service is used.

---

## Tests and verification

Run from the repository root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app core db scripts tests
git diff --check
```

Available workflow verification scripts include:

```bash
.venv/bin/python scripts/verify_demo_assets.py
.venv/bin/python scripts/verify_streamlit_workflow.py
.venv/bin/python scripts/evaluate_hospital_ranking.py
.venv/bin/python scripts/run_hospital_suggestions.py
```

The latest PR validation report records **327 passed, 14 warnings in 61.70 seconds**, along with successful compilation, whitespace checks, demo-asset verification, six-tab Streamlit AppTest workflow, and seven workflow scripts. These are the reported results for that tested PR revision; rerun the commands on any later checkout.

The full accident-model audit script requires its original cached feature dataset and associated frame metadata. If those artifacts are absent, the cache-dependent evaluation cannot be rerun from this checkout. Do not fabricate a cache or claim a fresh evaluation.

See `docs/FEATURE_COMPLETION_REPORT.md` and the PR description for the detailed implementation and validation record.

---

## Known limitations

- The accident classifier was trained on forward-facing dashcam video frames. Arbitrary still photographs and other camera viewpoints are out of domain. Its score is uncalibrated; false positives and false negatives occur.
- A low model score does not prove that no accident occurred. A high score does not prove that an accident occurred.
- A single still image cannot establish temporal impact. Two-dimensional bounding-box overlap is a spatial cue, not proof of physical contact.
- OCR may fail on small, blurred, angled, occluded, or synthetic plates. A format-valid candidate is not proof of a genuine registration.
- Caller transcripts, incident locations, and ambulance movements may be simulated. The project does not provide live GPS, live traffic, confirmed bed availability, hospital admission acceptance, or real ambulance dispatch.
- OpenStreetMap source records may be incomplete. Unknown values remain NOT DETERMINED; ranking is not clinical verification.
- The RAG corpus contains curated summaries, not full downloaded legal or policy documents. Its local TF-IDF retrieval is lexical, not a dense semantic embedding index. Answers are limited to the retrieved passages; the assistant is not medical or legal advice.
- Without a separately configured Ollama model, the response mode is extractive rather than LLM-generated.

---

## Project architecture decision summary

| Capability | Current status |
|---|---|
| Computer Vision | Implemented |
| Deep Learning | Implemented |
| Machine Learning | Implemented |
| OCR | Implemented |
| NLP | Implemented |
| Deterministic hospital retrieval | Implemented |
| Geospatial route simulation | Implemented |
| Deterministic hospital ranking | Implemented (independent of RAG) |
| Emergency-protocol knowledge RAG | Implemented (TF-IDF retrieval; optional local Ollama generation) |
| FAISS / SentenceTransformers hospital retrieval | Not used |
