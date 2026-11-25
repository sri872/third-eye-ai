# Third Eye AI — RAG Knowledge Assistant

## Purpose

The RAG assistant is a separate knowledge feature embedded in Tab 5, **Hospitals & Emergency Numbers**. It answers a small set of road-accident response and Good Samaritan questions using a curated, source-attributed local corpus.

It does **not** participate in hospital eligibility, hospital ranking, accident classification, incident assessment, or dispatch decisions. Hospital discovery remains deterministic and explainable in `core/hospitals/`.

## Architecture

1. Curated source summaries are stored in `data/knowledge_base/emergency_response_india.json`.
2. `core/rag/retriever.py` builds a local TF-IDF index with scikit-learn and retrieves relevant passages.
3. `core/rag/generator.py` produces an evidence-only extractive answer by default. If a local Ollama model is configured, it can generate a concise answer using only retrieved passages and inline source numbers.
4. `core/rag/pipeline.py` joins retrieval, answer generation, source references, and safety limitations.
5. Tab 5 exposes preset questions through a dropdown. The app does not need a paid API or an additional Python dependency.

## Generation modes

- **EXTRACTIVE RAG — NO LOCAL LLM CONFIGURED:** the response is assembled from retrieved passages; no LLM is claimed.
- **LOCAL OLLAMA RAG:** a locally running Ollama model generates a concise answer grounded in retrieved passages.
- **EXTRACTIVE FALLBACK — LOCAL OLLAMA UNAVAILABLE:** the local LLM call failed, so the app falls back to retrieved source text.

To enable optional local LLM generation, install/run Ollama separately, pull a compatible local model, and set:

```bash
export THIRD_EYE_RAG_OLLAMA_MODEL="qwen2.5:3b"
export THIRD_EYE_RAG_OLLAMA_HOST="http://127.0.0.1:11434"
```

The code does not install Ollama or download a model. When the environment variables are unset, the assistant works in extractive mode with no external service.

## Source corpus

The bundled corpus contains paraphrased summaries and metadata—not copied full documents—from:

- Ministry of Road Transport and Highways: Good Samaritan Rights and Duties.
- Government of India: Motor Vehicles (Amendment) Act, 2019, Section 134A.
- Government of India: Central Motor Vehicles (Twelfth Amendment) Rules, 2020.
- National Health Mission: Emergency Response Services / Patient Transport Service (108/102).
- WHO and UN Regional Commissions: Global Plan for the Decade of Action for Road Safety 2021–2030.
- Ministry of Road Transport and Highways: Scheme Guidelines on Rah-Veer (2025).

The corpus stores the official source URL with every document. Review source documents for current legal or operational details.

## Safety and scope

The assistant is informational only. It is not medical advice, legal advice, clinical triage, emergency dispatch, proof of hospital capability, or confirmation of ambulance availability. It does not diagnose injuries or recommend treatment. If there is an immediate emergency, contact the appropriate official emergency service directly; do not delay for the app.

Unsupported questions receive an explicit “not enough information in indexed sources” response. The assistant must not invent a citation or answer a question beyond the retrieved source material.

## Tests

```bash
.venv/bin/python -m pytest tests/test_rag_knowledge.py -q
```
