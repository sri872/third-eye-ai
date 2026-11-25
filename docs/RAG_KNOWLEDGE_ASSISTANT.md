# Third Eye AI — RAG Knowledge Assistant

## Purpose

The RAG assistant is a separate knowledge feature embedded in Tab 5, **Hospitals & Emergency Numbers**. It answers selected road-accident response and Good Samaritan questions using a curated, source-attributed local corpus.

It does **not** participate in hospital eligibility, hospital ranking, accident classification, incident assessment, or dispatch decisions. Hospital discovery remains deterministic and explainable in `core/hospitals/`.

## Architecture

1. Curated, paraphrased source summaries are stored in `data/knowledge_base/emergency_response_india.json`.
2. `core/rag/retriever.py` builds a local TF-IDF term-weight index with scikit-learn and retrieves relevant passages.
3. `core/rag/generator.py` uses an optional local Ollama model to write a concise answer from retrieved passages with inline passage-number citations.
4. If Ollama is not configured or unavailable, the system returns a labelled extractive evidence summary. This is a transparent RAG fallback, not an LLM-generated response.
5. `core/rag/pipeline.py` joins retrieval, answer generation/fallback, source references, and safety limitations.
6. Tab 5 exposes preset questions through a dropdown. No paid API key or additional Python dependency is required.

## Retrieval design

The retriever uses a local scikit-learn TF-IDF index with unigrams and bigrams, then ranks matching corpus passages by cosine similarity. This is lexical vector retrieval, not a dense embedding model or semantic neural retriever. Queries with no sufficiently relevant indexed passages receive an explicit no-supported-source response.

The hospital package in `core/hospitals/` is not connected to this retrieval index. It continues to use deterministic source-tag, facility-category, and proximity ranking.

## Generation modes

- **EXTRACTIVE RAG — NO LOCAL LLM CONFIGURED:** the response is assembled from retrieved passages; no LLM is claimed.
- **LOCAL OLLAMA RAG:** a locally running Ollama model generates a concise answer grounded in retrieved passages.
- **EXTRACTIVE FALLBACK — LOCAL OLLAMA UNAVAILABLE:** the local LLM call failed, so the app falls back to retrieved source text.
- **NO SUPPORTED SOURCE FOUND:** no sufficiently relevant passage was retrieved and no answer is invented.

To enable optional local LLM generation, install/run Ollama separately, pull a compatible local model, and set:

```bash
export THIRD_EYE_RAG_OLLAMA_MODEL="qwen2.5:3b"
export THIRD_EYE_RAG_OLLAMA_HOST="http://127.0.0.1:11434"
```

The code does not install Ollama or download a model. When the environment variables are unset, the assistant works in extractive mode with no external service.

## Source corpus

The bundled corpus contains paraphrased summaries and metadata—not copied full documents—from:

- Ministry of Road Transport and Highways: Good Samaritan Rights and Duties of Hospitals/Police.
- Government of India: Motor Vehicles (Amendment) Act, 2019, Section 134A.
- National Health Mission: Emergency Response Services / Patient Transport Service (108/102).
- WHO and UN Regional Commissions: Global Plan for the Decade of Action for Road Safety 2021–2030.

Each document record contains its official source URL and agency. Review the original source documents for current legal or operational details. The corpus is a curated prototype knowledge set, not an exhaustive legal or emergency-protocol library.

## Safety and scope

The assistant is informational only. It is not medical advice, legal advice, clinical triage, emergency dispatch, proof of hospital capability, or confirmation of ambulance availability. It does not diagnose injuries or recommend treatment. If there is an immediate emergency, contact the appropriate official emergency service directly; do not delay for the app.

Unsupported questions receive an explicit “not enough information in indexed sources” response. The assistant must not invent a citation or answer a question beyond the retrieved source material. Retrieval relevance scores are lexical ranking signals, not answer confidence.

## Tests

```bash
.venv/bin/python -m pytest tests/test_rag_knowledge.py tests/test_hospitals_architecture.py -q
```
