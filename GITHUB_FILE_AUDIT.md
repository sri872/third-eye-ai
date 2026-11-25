# GitHub File Audit — Third Eye AI

> **Inventory supersession (2026-11-25):** This file inventory predates the hospital retrieval cleanup. FAISS and SentenceTransformers are removed from the current branch's `requirements.txt` and runtime code. Hospital suggestions use deterministic source-backed geospatial ranking. See [README.md](README.md) and [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md) for the current architecture.


**Repository:** <https://github.com/sri872/third-eye-ai> (public)
**Workspace:** `~/Projects/third-eye-v2`
**Audit date:** 2026-11-25
**Initial release commit:** `ecbbeda` (`Initial release: THIRD EYE AI`)

This document is the complete inventory and publication checklist for the GitHub repository. It records what is committed, what is deliberately excluded, how large assets are handled, and the secret-scan result.

---

## 1. Tracked repository summary

At the initial release commit, Git tracked **158 files** totalling **132,179,291 bytes (≈126.0 MiB)**. No single tracked file exceeds GitHub's 100 MB hard file limit, and the repository is well below GitHub's 1 GB soft repository-size warning threshold.

| Top-level path | Tracked files | Contents |
|---|---:|---|
| `data/` | 40 | Demo media, sample calls, cached OSM records, route GeoJSON/PNG outputs, committed verification JSON |
| `core/` | 33 | CV, OCR, ML, NLP, RAG and routing implementation |
| `tests/` | 26 | Unit, integration, UI and demo-regression tests |
| `docs/` | 14 | Phase reports, demo script, model/system cards |
| `scripts/` | 12 | Training, verification and offline analysis utilities |
| `models/` | 7 | EasyOCR weights, YuNet ONNX + licence, trained classifier and metrics |
| `app/` | 7 | Streamlit entry point and tab helpers |
| `config/` | 3 | Demo locations, region and simulated ambulance configuration |
| `db/` | 2 | SQLite schema helpers and audit tests |
| Root | 15 | `README.md`, reports, `.gitignore`, `.env.example`, `requirements.txt`, `pytest.ini`, `launch.command` |

File types: 80 `.py`, 23 `.md`, 16 `.txt`, 10 `.json`, 6 `.jpg`, 4 `.png`, 4 `.geojson`, 3 `.jpeg`, 2 `.pth`, 1 each `.osm`, `.onnx`, `.mp4`, `.joblib`, `.ini`, `.graphml`, `.gitignore`, `.env.example`, `.command`.

### Documentation committed at the root

- `README.md` — installation, launch, six-technology map, six-tab guide, professor demo sequence
- `FINAL_DEMO_VERIFICATION_REPORT.md` — authoritative measured outputs and limitations
- `GITHUB_FILE_AUDIT.md` — this inventory
- `FIVE_TECHNOLOGY_AUDIT_REPORT.md`, `NLP_AUDIT_REPORT.md` — technology audit appendices
- `DEMO_READINESS_REPORT.md`, `PHASE12_REPORT.md` … `PHASE15_FINAL_VERIFICATION.md` — historical phase evidence

No root `LICENSE` is included because no licence has been selected by the project owner. The vendored YuNet model retains its own Apache-2.0 licence at `models/plate_detector/LICENSE`; no other licence is implied for the surrounding project.

---

## 2. Large and important assets

| Asset | Bytes | SHA-256 | Handling |
|---|---:|---|---|
| `models/easyocr/craft_mlt_25k.pth` | 83,152,330 | `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17` | Committed directly; ~79 MiB, below GitHub's 100 MB limit. |
| `models/easyocr/english_g2.pth` | 15,143,997 | `e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8` | Committed directly. |
| `data/samples/demo/road vdo.mp4` | 14,323,936 | `cdc01a90b62616b62abeeaa8110de9314a60c4dce0d140820cbd7d54c7a76809` | Committed directly; mandatory demo video. |
| `data/phagwara_corridor.osm` | 5,514,700 | — | Committed; offline OSM source cache. |
| `data/phagwara_drive_network.graphml` | 4,015,834 | — | Committed; offline routing graph. |
| `models/plate_detector/license_plate_detection_lpd_yunet_2023mar.onnx` | 4,146,213 | `6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7` | Committed; exact hash checked at load time. |
| `models/accident_classifier.joblib` | ~1 KB | `672f5fc05909bab42c85e70718f44bb4d15431724fbf4984b212350fb0e967a4` | Committed with `models/accident_classifier_metrics.json`. |

Git LFS is **not** used. The largest object is under the hard limit, so a normal clone works without LFS credentials or a separate fetching step.

Assets downloaded once by their libraries and stored outside the repository:

- TorchVision Faster R-CNN ResNet-50 FPN COCO weights — `~/.cache/torch/hub/checkpoints`
- TorchVision ResNet-18 weights — same TorchVision cache
- SentenceTransformer `all-MiniLM-L6-v2` — Hugging Face cache; after the first download the runtime uses offline mode

These external caches are not committed and are documented in `README.md`.

### Mandatory demo input hashes

| File | Bytes | SHA-256 |
|---|---:|---|
| `single_car_plate.jpg` | 114,036 | `246cf1cd4dc1cdc58592c4b92878289c70a71c051039d3c71e4a417322fb612b` |
| `two_car_collision.jpg` | 312,179 | `3fdb9739daeba222e2782aabf02952577ee3a3dc6a81b3a6ba7883e02009c3c9` |
| `heavy_traffic.jpg` | 172,786 | `eb25eb9229002de506d4c622bd43793c461b6ddeb51afb5f4605d25be77967b9` |
| `road vdo.mp4` | 14,323,936 | `cdc01a90b62616b62abeeaa8110de9314a60c4dce0d140820cbd7d54c7a76809` |

`img1.jpeg`, `img2.jpeg` and the WhatsApp-named JPEG under `data/samples/demo/` are byte-identical aliases of the canonical files above. They are retained for traceability to the original source names; the professor demonstration uses only the four canonical filenames.

---

## 3. Deliberately excluded files

`.gitignore` excludes:

| Pattern | Reason |
|---|---|
| `.venv/`, `venv/`, `ENV/` | Local Python environments; recreated from `requirements.txt`. |
| `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.coverage`, `htmlcov/` | Generated Python/test artefacts. |
| `*.db`, `*.sqlite`, `*.sqlite3`, `db/third_eye.db` | Runtime/state databases; schema is recreated by `db.database.init_db()`. |
| `data/frames/`, `data/embeddings.*` | Training-only caches; regenerate with the frame extractor and training script. |
| `data/outputs/*` except `dispatch_summary.json`, `demo_verification.json`, `streamlit_workflow.json`, and `routes/` | Transient crops/annotated frames; the four whitelisted evidence files are required for the report and Dispatch replay. |
| `data/outputs/crops/` | Large generated plate/vehicle crops (~25 MB in the working tree); reproducible from uploaded media. |
| `.env`, `.env.*` except `.env.example` | Local secrets/overrides; no secret is needed by the app. |
| `*.pem`, `*.key`, `id_rsa*`, `*.p12` | Private keys. |
| `.DS_Store`, `.idea/`, `.vscode/`, `.streamlit/` | Editor/OS/local Streamlit state. |
| `*.log`, `*.tmp`, `*.swp` | Temporary files. |

The app never writes uploaded bytes into the project. It uses in-memory bytes for images and an auto-deleted temporary file outside the project for videos; SQLite is opened read-only (`mode=ro`).

---

## 4. Secret and credential scan

A recursive text scan of the working tree, excluding `.git`, `.venv`, bytecode caches and binary model/media files, searched for:

- OpenAI-style `sk-…` keys
- GitHub personal access tokens (`ghp_…`, `github_pat_…`)
- AWS access-key IDs (`AKIA…`)
- PEM/OpenSSH/EC/RSA private-key headers
- Hardcoded `api_key`/`api-key` assignments with long quoted values

**Result:** 0 matches. No credential is committed. `.env.example` contains only an optional dispatch-summary path override and explicitly states that no API key is required.

---

## 5. Publication checklist

- [x] Public repository created at `https://github.com/sri872/third-eye-ai`
- [x] `main` branch and `origin` remote configured
- [x] Initial release commit `ecbbeda` pushed to `origin/main`
- [x] All four mandatory demo inputs and their hashes recorded
- [x] Model weights and OSM caches present in the tracked tree or documented external caches
- [x] Generated crops, databases, environments and secrets excluded
- [x] README, final verification report and this audit present
- [x] Full test suite and real-inference verification completed (see `FINAL_DEMO_VERIFICATION_REPORT.md`)
- [x] Final follow-up commit for post-release fixes and new evidence JSON pushed to `origin/main`

After the final follow-up commit, verify with:

```bash
git status --short
git log --oneline -3
git push origin main
gh repo view sri872/third-eye-ai --json name,visibility,url
```
