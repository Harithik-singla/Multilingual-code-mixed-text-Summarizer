# 🧠 Project Context — Read This First (Agent Handoff Doc)

> This file exists so a future AI agent can instantly understand the full project
> without the user needing to re-explain everything. Read this entire file before
> making any suggestions or writing any code.
>
> **Last updated: 2026-04-13**

---

## 📌 Project Identity

- **Project name:** Multilingual Code-Mixed Text Summarization
- **Local path:** `c:\Users\rauna\Videos\Code mixed text summarisation\`
- **Author:** Raunaq Mittal (`raunaqmittal2004@gmail.com`)
- **Purpose:** College-level project / portfolio piece
- **Timeline:** 1 month
- **Code style:** Keep it SIMPLE. No complex abstractions, no type hints, no argparse. College-level readable code.

---

## 🎯 What This Project Does

Builds a **transformer-based abstractive text summarization system** that:
- Accepts **code-mixed multilingual input** (Hinglish, Banglish, Gujarati-English)
- Works WITHOUT any external translation APIs
- Generates concise summaries
- Uses **IndicBART** (`ai4bharat/IndicBART`) as the backbone model

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
    ├── hindi_english.json     (117 words)
    ├── bengali_english.json   (117 words)
    └── gujarati_english.json  (117 words)
```

---

## 🏋️ Training Strategy (3 Phases)

### Phase 1: Monolingual Training
- Data: ALL 4 ILSUM languages combined + shuffled (not per-language sequential)
- Config: `src/configs/phase1_train.yaml` — 5 epochs, lr=3e-5, batch=2, grad_accum=8, fp16
- Checkpoint: `artifacts/checkpoints/phase1/`
- Status: **RUNNING** on full dataset (GPU configured)

### Phase 2: Code-Mixed Adaptation
- Data: 70% synthetic + 30% real HinGE
- Config: `src/configs/phase2_train.yaml` — loads from phase1 checkpoint
- Checkpoint: `artifacts/checkpoints/phase2/`
- Status: **IMPLEMENTED** (Code ready, waiting for Phase 1 to finish)

### Phase 3: Stabilization
- Data: Phase 2 data + 20% ILSUM mix
- Config: `src/configs/phase3_train.yaml` — loads from phase2 checkpoint
- Checkpoint: `artifacts/checkpoints/phase3/`
- Status: **IMPLEMENTED** (Code ready, waiting for Phase 2 to finish)

---

## 📁 Complete File Status

### ✅ FULLY IMPLEMENTED (Simple, clean code)

| File | Lines | What it does |
|------|-------|-------------|
| `src/components/data/dataset_loader.py` | 128 | Downloads ILSUM + HinGE, splits 70/15/15, saves CSVs |
| `src/components/data/preprocessor.py` | 158 | Cleans text (URLs, HTML, control chars), adds `<2xx>` language tags |
| `src/components/data/synthetic_generator.py` | 191 | Loads dicts from JSON, replaces 20-40% words with English |
| `src/components/model/model_loader.py` | 78 | Loads IndicBART model + tokenizer, detects GPU/CPU |
| `src/components/model/tokenizer_utils.py` | 153 | Adds language tag tokens, tokenizes DataFrames for training |
| `src/components/model/trainer.py` | 183 | Wraps Seq2SeqTrainer, reads YAML config, computes ROUGE |
| `src/pipelines/train_phase1.py` | 125 | Loads all ILSUM, combines, tokenizes, trains Phase 1 |
| `src/pipelines/train_phase2.py` | 108 | Phase 2 adaptation pipeline (70% synthetic/30% real) |
| `src/pipelines/train_phase3.py` | 114 | Phase 3 stabilization pipeline (Phase2 mix + 20% ILSUM) |
| `src/components/inference/language_detector.py` | 46 | Auto-detects dominant language via Unicode ranges |
| `src/components/inference/predictor.py` | 84 | End-to-end inference using best checkpoint |
| `src/exception/exception.py` | 22 | Custom exception with file + line info |
| `src/logging/logger.py` | 16 | Timestamped file logging to `src/logs/` |
| `setup.py` | 43 | Package definition |
| All YAML configs | — | model_config, phase1/2/3_train, inference_config |

### ❌ STUB ONLY (docstring, no code)

| File | What it should do |
|------|-------------------|
| `src/components/data/transliterator.py` | Optional: convert Romanized Hindi → Devanagari |
| `src/components/evaluation/rouge_scorer.py` | Standalone ROUGE evaluation |
| `src/components/evaluation/bert_scorer.py` | BERTScore evaluation |
| `src/pipelines/generate_synthetic.py` | Standalone synthetic generation script |
| `src/pipelines/evaluate.py` | Evaluation pipeline |
| `src/pipelines/inference_pipeline.py` | CLI inference |
| `app.py` | FastAPI/Gradio web demo |
| `tests/test_preprocessor.py` | Unit tests |
| `tests/test_synthetic_gen.py` | Unit tests |
| `tests/test_predictor.py` | Integration test |

---

## 🔗 How Files Connect (Data Flow)

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

### train_phase1.py connects the pieces like this:
1. `load_combined_split("train")` → reads `preprocessed/[Lang]/train_clean.csv` for all 4 languages, combines + shuffles
2. `ModelLoader().load()` → returns `(model, tokenizer, device)`
3. `TokenizerUtils(tokenizer)` → adds language tag special tokens, provides `tokenize_dataframe()`
4. `ModelTrainer(phase=1, ..., vocab_size=tok.vocab_size())` → resizes embeddings, builds Seq2SeqTrainer
5. `trainer.train(train_dataset, val_dataset)` → runs training, saves checkpoint

---

## ▶️ How to Run

```bash
# Step 1: Download + split data (already done — skip)
python -m src.components.data.dataset_loader

# Step 2: Preprocess data (already done — skip)
python -m src.components.data.preprocessor

# Step 3: Generate synthetic data (already done — skip)
python -m src.components.data.synthetic_generator

# Step 4: Train Phase 1 (CURRENTLY RUNNING)
python -m src.pipelines.train_phase1

# Step 5: After Phase 1, run Phase 2 & 3
python -m src.pipelines.train_phase2
python -m src.pipelines.train_phase3
```

---

## ⚠️ Known Issues

1. **Romanized Hinglish:** Text like `"aur bhai kya kar rha"` (Hindi in Latin script) is NOT Devanagari. The synthetic generator skips these. HinGE dataset naturally contains Romanized Hinglish — model learns it via Phase 2.

2. **GPU setup:** System has RTX 3050 Ti (4GB VRAM). PyTorch CUDA build installed from local WHL: `torch-2.6.0+cu124-cp313-cp313-win_amd64.whl`. Verified working: `torch.cuda.is_available() = True`.

3. **bf16 instead of fp16:** `trainer.py` uses **bf16** (not fp16). fp16 caused `grad_norm: nan` due to overflow. bf16 has the same exponent range as fp32 so it is NaN-safe, and RTX 3050 Ti (Ampere) supports bf16 natively. `mixed_precision: true` in any YAML config now automatically maps to bf16.

4. **max_input_length: 300** (reduced from 512). News articles front-load key information so 300 tokens captures enough content. Provides ~35% speed improvement with negligible accuracy loss.

5. **predict_with_generate per phase:** Phase 1 skips beam-search generation during eval (faster). Phase 2 and 3 run full ROUGE evaluation after each epoch.

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
4. **Settings as plain variables at the top of the file** — no argparse
5. **Each component has a `run()` or `main()` function** as the entry point
6. **Each file has `if __name__ == "__main__"` block** for standalone testing
7. **Artifact paths** — always relative to project root: `artifacts/data/`, `artifacts/checkpoints/`

---

## 🔨 What to Build Next (in order)

```
1. inference_pipeline.py  ← CLI wrapper around Predictor
2. evaluate.py            ← Evaluation orchestrator
3. rouge_scorer.py        ← Standalone ROUGE using evaluate.py
4. bert_scorer.py         ← Standalone BERTScore using evaluate.py
5. app.py                 ← Web demo (Gradio)
6. Tests
```
