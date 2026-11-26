# Third Eye AI — System Architecture

## 1. Overview
Third Eye AI is a focused, explainable incident analysis prototype combining three core deterministic disciplines:
1. **Classical Computer Vision (CV)** — Deterministic image processing, enhancement, edge detection, and geometric vehicle candidate localization using OpenCV and NumPy.
2. **Optical Character Recognition (OCR)** — Classical Indian registration plate localization, character segmentation, template matching against standardized character glyphs, and statutory format validation without neural networks or external OCR engines.
3. **Natural Language Processing (NLP)** — Deterministic regular-expression and keyword-based incident extraction, vehicle categorization, injury/hazard tagging, urgency classification, and multi-source structured reporting.

All Machine Learning (ML), Deep Learning (DL), PyTorch, TorchVision, YOLO, ResNet, EasyOCR, neural ONNX models, and Retrieval-Augmented Generation (RAG) subsystems have been completely removed.

---

## 2. Target Directory Structure

```text
THIRD-EYE-AI/
├── app/
│   ├── main.py                    # Streamlit dashboard entrypoint
│   └── components/
│       ├── cv_ui.py               # Section 1: CV image processing & candidate detection
│       ├── ocr_ui.py              # Section 2: Indian plate template-based OCR
│       ├── nlp_ui.py              # Section 3: NLP incident information extraction
│       └── report_ui.py           # Section 4: Combined multi-source incident report
│
├── core/
│   ├── cv/
│   │   ├── image_processing.py    # Grayscale, CLAHE, blur, Canny, diagnostics
│   │   ├── vehicle_localization.py# Edge density & contour morphology candidates
│   │   ├── plate_localization.py  # Top-hat, Sobel gradients, aspect-ratio bounds
│   │   └── visualization.py       # Box drawing & annotated canvas overlays
│   │
│   ├── ocr/
│   │   ├── preprocessing.py       # Plate deskewing, bilateral filtering, binarization
│   │   ├── character_segmentation.py # Connected-component / contour glyph segmentation
│   │   ├── template_recognition.py# Alphanumeric template matching with positional priors
│   │   ├── plate_validation.py    # Indian Motor Vehicles format validation (HSRP/BH)
│   │   └── plate_reader.py        # End-to-end classical plate reading coordinator
│   │
│   └── nlp/
│       ├── text_normalization.py  # Typo correction, whitespace & token normalization
│       ├── entity_extraction.py   # Deterministic entity & keyword extraction
│       ├── incident_analysis.py   # Incident type classification & urgency rules
│       └── report_generation.py   # Multi-source structured report synthesis
│
├── data/
│   ├── samples/                   # Sample road images and collision photos
│   ├── sample_calls/              # Emergency transcript presets
│   └── ocr_templates/             # 24x32 standard binary character templates (A-Z, 0-9)
│
├── tests/
│   ├── test_cv.py                 # CV enhancement, diagnostics, localization tests
│   ├── test_ocr.py                # Plate preprocessing, template matching, format tests
│   ├── test_nlp.py                # Regex extraction, urgency, contradiction tests
│   └── test_app.py                # Integration and Streamlit component tests
│
├── docs/
│   ├── ARCHITECTURE.md            # System architecture and design
│   ├── CV_DOCUMENTATION.md        # Computer Vision pipeline details
│   ├── OCR_DOCUMENTATION.md       # Template OCR & plate validation details
│   └── NLP_DOCUMENTATION.md       # Rule-based NLP extraction & urgency details
│
├── requirements.txt               # Minimal dependencies: streamlit, opencv-python, numpy, pillow, pytest
└── README.md                      # Comprehensive user and demonstration documentation
```

---

## 3. Technology Stack and Boundaries

| Technology Area | Permitted Implementation | Strictly Prohibited |
|---|---|---|
| **Computer Vision** | OpenCV (v4.8+), NumPy array math, Canny edges, Otsu/adaptive thresholding, morphological closing/opening, contour analysis, bounding box geometry | YOLO, Faster R-CNN, ResNet, CNNs, ViT, Haar cascades, HOG/SVM, pretrained neural networks |
| **OCR** | Top-hat/black-hat morphology, Sobel gradients, Hough line deskewing, connected-component segmentation, 24x32 normalized correlation + IoU template matching | EasyOCR, Tesseract, CRAFT, CRNN, Transformer OCR, neural character classifiers |
| **NLP** | Python standard library `re`, string parsing, keyword dictionaries, transparent urgency decision trees | spaCy models, HuggingFace transformers, BERT, LLMs, Ollama, vector databases, dense embeddings |
| **Interface** | Streamlit single-page application with 4 focused tabs | Unnecessary routing maps, hospital dispatch simulation, live GPS tracking |
