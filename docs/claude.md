# 🧠 Project Context — Read This First (Agent Handoff Doc)

> This file exists so a future AI agent can instantly understand the full project
> without the user needing to re-explain everything. Read this entire file before
> making any suggestions or writing any code.
>
> **Last updated: 2026-04-14**

---

## 📌 Project Identity

- **Project name:** Multilingual Code-Mixed Text Summarization
- **Local path:** `c:\Users\rauna\Videos\Code mixed text summarisation\`
- **Author:** Raunaq Mittal (`raunaqmittal2004@gmail.com`)
- **Purpose:** College-level project / portfolio piece
- **Timeline:** 1 month
- **Code style:** Keep it SIMPLE. No complex abstractions, no type hints, no argparse (except CLI tools). College-level readable code.

---

## 🎯 What This Project Does

Builds a **transformer-based abstractive text summarization system** that:
- Accepts **code-mixed multilingual input** (Hinglish, Banglish, Gujarati-English)
- Works WITHOUT any external translation APIs at inference time
- Generates concise summaries
- Uses **IndicBART** (`ai4bharat/IndicBART`) as the backbone model
- Uses **Streamlit** for the web demo frontend

---

## 🤖 Model

- **Model:** `ai4bharat/IndicBART` (~244M params, seq2seq)
- **Language tags (prepended to input text ONLY, never to summary):**
  - Hindi → `<2hi>`, Bengali → `<2bn>`, Gujarati → `<2gu>`, English → `<2en>`
  - Hinglish → `<2hi>` (Hindi is dominant)
- **Config file:** `src/configs/model_config.yaml`

---

## 📊 Datasets

### 1. ILSUM-2.0 (`ILSUM/ILSUM-2.0`)
- Monolingual Indian language news articles + summaries
- Languages: Hindi, Bengali, English, Gujarati
- Column mapping: `Article → text`, `Summary → summary`
- Used for: **Phase 1 training**

### 2. HinGE (`LingoIITGN/HinGE`)
- Real Hinglish (code-mixed) dataset
- Column mapping: `Human-generated Hinglish → text`, `English → summary`
- Used for: **Phase 2 training** (30% of data mix)

---

## 📁 Data on Disk (ALL EXISTS — DO NOT RE-RUN DATA PIPELINE)

```
artifacts/data/
├── ilsum_processed/           ← raw downloads split 70/15/15
│   ├── Hindi/                 train.csv (~128MB), val.csv, test.csv
│   ├── Bengali/               train.csv, val.csv, test.csv
│   ├── English/               train.csv, val.csv, test.csv
│   └── Gujarati/              train.csv, val.csv, test.csv
├── preprocessed/              ← cleaned + language tags added
│   ├── Hindi/                 train_clean.csv, val_clean.csv, test_clean.csv
│   ├── Bengali/               ...
│   ├── English/               ...
│   ├── Gujarati/              ...
│   └── real_codemixed/        train_clean.csv, val_clean.csv, test_clean.csv
├── synthetic_codemixed/       ← 20-40% words replaced with English
│   ├── Hindi/                 train_synthetic.csv, val_synthetic.csv, test_synthetic.csv
│   ├── Bengali/               ...
│   └── Gujarati/              ...
├── real_codemixed/            ← HinGE raw splits
│   └── train.csv (~484KB), val.csv, test.csv
└── dictionaries/              ← JSON word maps (Hindi/Bengali/Gujarati → English)
    ├── hindi_english.json     (265 words)
    ├── bengali_english.json   (117 words)
    └── gujarati_english.json  (117 words)
```

---

## 🏋️ Training Strategy (3 Phases) — ALL COMPLETE ✅

### Phase 1: Monolingual Training ✅
- Data: ALL 4 ILSUM languages combined + shuffled
- Config: `src/configs/phase1_train.yaml` — 5 epochs, lr=3e-5, batch=16, grad_accum=1, bf16
- Checkpoint: `artifacts/checkpoints/phase1/`

### Phase 2: Code-Mixed Adaptation ✅
- Data: 70% synthetic (Hindi+Bengali+Gujarati) + 30% real HinGE
- Config: `src/configs/phase2_train.yaml` — loads from phase1 checkpoint
- Checkpoint: `artifacts/checkpoints/phase2/`

### Phase 3: Stabilization ✅
- Data: Phase 2 data + 20% ILSUM mix
- Config: `src/configs/phase3_train.yaml` — loads from phase2 checkpoint
- Checkpoint: `artifacts/checkpoints/phase3/`

> **Note on training output:** The ILSUM dataset summaries are bilingual (English headline + Hindi/Bengali/Gujarati text). The HinGE dataset summaries are pure English. This means model output is inconsistent — Hinglish input tends to generate English summaries, while Devanagari Hindi input generates bilingual summaries. This is a known characteristic, not a bug.

---

## 🔁 Transliterator Strategy

The transliterator is used **only during inference** (not during training):
- **Hindi/Hinglish input →** Transliterator is **SKIPPED**. The model already handles Romanized Hinglish natively because it was trained on the HinGE dataset.
- **Bengali/Gujarati Romanized input →** Transliterator converts the text to native script (Bengali/Gujarati Unicode) before feeding it to the model.
- Uses `indic-transliteration` library (already in `requirements.txt`).
- Integrated into `predictor.py` — runs automatically based on detected language tag.

---

## 📁 Complete File Status

### ✅ ALL FILES FULLY IMPLEMENTED

| File | Lines | What it does |
|------|-------|-------------|
| `src/components/data/dataset_loader.py` | 128 | Downloads ILSUM + HinGE, splits 70/15/15, saves CSVs |
| `src/components/data/preprocessor.py` | 158 | Cleans text (URLs, HTML, control chars), adds `<2xx>` language tags |
| `src/components/data/synthetic_generator.py` | 191 | Loads dicts from JSON, replaces 20-40% words with English |
| `src/components/data/transliterator.py` | 40 | Converts Romanized Bengali/Gujarati to native script (skips Hindi) |
| `src/components/model/model_loader.py` | 78 | Loads IndicBART model + tokenizer, detects GPU/CPU |
| `src/components/model/tokenizer_utils.py` | 152 | Adds language tag tokens, tokenizes DataFrames for training |
| `src/components/model/trainer.py` | 188 | Wraps Seq2SeqTrainer, reads YAML config, computes ROUGE |
| `src/components/inference/language_detector.py` | 51 | Auto-detects dominant language via Unicode ranges |
| `src/components/inference/predictor.py` | 87 | End-to-end inference with transliterator integration |
| `src/components/evaluation/rouge_scorer.py` | 36 | ROUGE-1/2/L F-measure computation |
| `src/components/evaluation/bert_scorer.py` | 33 | BERTScore precision/recall/F1 |
| `src/pipelines/train_phase1.py` | 125 | Phase 1 monolingual training pipeline |
| `src/pipelines/train_phase2.py` | 110 | Phase 2 code-mixed adaptation pipeline |
| `src/pipelines/train_phase3.py` | 125 | Phase 3 stabilization pipeline |
| `src/pipelines/inference_pipeline.py` | 30 | CLI wrapper around Predictor |
| `src/pipelines/evaluate.py` | 70 | Full evaluation pipeline, saves JSON results |
| `app.py` | 55 | Streamlit web demo |
| `tests/test_preprocessor.py` | 35 | Unit tests for text cleaning |
| `tests/test_synthetic_gen.py` | 38 | Unit tests for synthetic generation |
| `tests/test_predictor.py` | 35 | Integration tests (auto-skip if no checkpoint) |
| All YAML configs | — | model_config, phase1/2/3_train, inference_config |
| `src/exception/exception.py` | 22 | Custom exception with file + line info |
| `src/logging/logger.py` | 16 | Timestamped file logging to `src/logs/` |
| `setup.py` | 43 | Package definition |

### ❌ NO STUBS REMAINING — All files are fully implemented.

---

## 🔗 How Files Connect (Data Flow)

### Training Flow
```
dataset_loader.py  →  preprocessor.py  →  synthetic_generator.py
                                                   ↓
train_phase1.py reads from:  preprocessed/[Lang]/train_clean.csv
                             ↓
                    model_loader.py  →  tokenizer_utils.py  →  trainer.py
                             ↓                    ↓                ↓
                    loads IndicBART      adds <2hi> tokens     Seq2SeqTrainer
                             ↓                    ↓                ↓
                                    artifacts/checkpoints/phase1/
```

### Inference Flow
```
User Input Text
    ↓
language_detector.py  →  detects <2hi>, <2bn>, or <2gu>
    ↓
transliterator.py  →  converts Bengali/Gujarati Romanized → native script
    ↓                  (skips Hindi — model handles Hinglish natively)
predictor.py  →  tokenize → model.generate() → decode
    ↓
Generated Summary
```

---

## ▶️ How to Run

```bash
# Data pipeline (ALREADY DONE — DO NOT RE-RUN)
python -m src.components.data.dataset_loader
python -m src.components.data.preprocessor
python -m src.components.data.synthetic_generator

# Training (ALREADY DONE — checkpoints on remote GPU)
python -m src.pipelines.train_phase1
python -m src.pipelines.train_phase2
python -m src.pipelines.train_phase3

# Quick CLI test
python -m src.pipelines.inference_pipeline --text "aaj bharat ne match jeeta"

# Full evaluation (saves to artifacts/results/eval_results.json)
python -m src.pipelines.evaluate

# Launch Streamlit web app
streamlit run app.py

# Run unit tests
python -m pytest tests/ -v
```

---

## ⚠️ Known Issues

1. **Bilingual Summaries:** For Devanagari Hindi input, the model outputs bilingual summaries (English headline + Hindi text) because that's the pattern in the ILSUM training data. This is by design, not a bug.

2. **Romanized Hinglish:** Text like `"aur bhai kya kar rha"` is handled natively via HinGE Phase 2 training. No transliteration needed for Hindi.

3. **Transliterator for Bengali/Gujarati only:** The `indic-transliteration` library handles Romanized → native script conversion. Only used when `language_detector` detects `<2bn>` or `<2gu>`.

4. **GPU setup:** Training was done on remote GPU (20GB VRAM). Configs: batch_size=16, grad_accum=1, bf16=True. For local inference, CPU works fine.

5. **bf16 instead of fp16:** `trainer.py` uses **bf16** (not fp16). fp16 caused `grad_norm: nan` due to overflow.

6. **max_input_length: 512, max_output_length: 256** — configured for the 20GB VRAM remote GPU.

---

## 📐 Code Conventions (MUST FOLLOW)

1. **Keep code SIMPLE** — college-level readable, no over-engineering
2. **Exception pattern:**
   ```python
   except Exception as e:
       raise CodeMixedSummarizationException(e, sys)
   ```
3. **Logging:**
   ```python
   from src.logging.logger import logging
   logging.info("message")
   ```
4. **Settings as plain variables at the top of the file** — no argparse (except CLI tools)
5. **Each component has a `run()` or `main()` function** as the entry point
6. **Each file has `if __name__ == "__main__"` block** for standalone testing
7. **Artifact paths** — always relative to project root: `artifacts/data/`, `artifacts/checkpoints/`
8. **Frontend:** Streamlit (NOT Gradio/FastAPI)
