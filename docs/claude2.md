# 🧠 Project Context — Read This First (Agent Handoff Doc)

> This file exists so a future AI agent can instantly understand the full project
> without the user needing to re-explain everything. Read this entire file before
> making any suggestions or writing any code.

---

## 📌 Project Identity

- **Project name:** Multilingual Code-Mixed Text Summarization (Indic Languages + English)
- **GitHub corpus:** `raunaqmittal/Multilingual-code-mixed-text-Summarization`
- **Local path:** `c:\Users\rauna\Videos\Code mixed text summarisation\`
- **Author:** Raunaq Mittal (`raunaqmittal2004@gmail.com`)
- **Purpose:** Internship / research portfolio project
- **Timeline:** 1 month

---

## 🎯 What This Project Does

Builds a **transformer-based abstractive text summarization system** that:
- Accepts **code-mixed multilingual input** (e.g., Hinglish, Banglish, Gujarati-English)
- Works WITHOUT any external translation APIs
- Generates concise English-or-native summaries
- Uses **IndicBART** (`ai4bharat/IndicBART`) as the backbone model

---

## 🤖 Model

- **Model:** `ai4bharat/IndicBART` (~244M params, seq2seq, multilingual)
- **Why:** Designed for Indian languages, handles script diversity, efficient
- **Language tags (IndicBART official format):**
  - Hindi → `<2hi>`
  - Bengali → `<2bn>`
  - Gujarati → `<2gu>`
  - English → `<2en>`
  - Hinglish → `<2hi>` (Hindi is dominant language)
- **Tag format:** `"<2hi> actual text here..."` — tag + single space + text
- **Tags are ONLY prepended to `text` (input) column, NEVER to `summary` (target)**

---

## 📊 Datasets

### 1. ILSUM-2.0 (`ILSUM/ILSUM-2.0`)
- Monolingual Indian language news articles + summaries
- Languages loaded: Hindi, Bengali, English, Gujarati
- Used for: **Phase 1 (monolingual training)**
- HuggingFace ID: `ILSUM/ILSUM-2.0`
- Column mapping from raw: `Article → text`, `Summary → summary`

### 2. HinGE (`LingoIITGN/HinGE`)
- Real Hinglish (code-mixed) dataset
- Column mapping: `Human-generated Hinglish → text` (code-mixed input), `English → summary` (clean output)
- Language tag: `Hinglish`
- Used for: **Phase 2 (real code-mixed adaptation)**
- Note: HinGE text is Romanized Hinglish (Latin script, not Devanagari)

---

## 🔧 Data Processing Pipeline (ALL COMPLETE — DATA IS ON DISK)

### Step 1: DatasetLoader (`src/components/data/dataset_loader.py`) ✅ DONE
- Loads ILSUM (4 languages) + HinGE from HuggingFace
- Splits data 70% train / 15% val / 15% test using `sklearn.train_test_split`
- Saves to `artifacts/data/ilsum_processed/[Language]/train,val,test.csv`
- HinGE saved to `artifacts/data/real_codemixed/train,val,test.csv`

### Step 2: TextPreprocessor (`src/components/data/preprocessor.py`) ✅ DONE
- NFC Unicode normalization
- Removes: URLs, HTML tags/entities, control characters, zero-width chars
- Normalizes whitespace
- Prepends language tag to `text` column only
- Saves to `artifacts/data/preprocessed/[Language]/train_clean.csv` etc.
- Also: `artifacts/data/preprocessed/real_codemixed/train_clean.csv`

### Step 3: SyntheticDataGenerator (`src/components/data/synthetic_generator.py`) ✅ DONE
- Takes ILSUM preprocessed (Hindi/Bengali/Gujarati — English SKIPPED)
- Replaces 20–40% of content words with English equivalents (dictionary-based)
- Built-in dictionaries: ~100 words per language (politics, economy, education, tech, sports)
- External JSON dicts saved to `artifacts/data/dictionaries/` override built-in
- Only `text` column gets mixed; `summary` stays clean
- Output: `artifacts/data/synthetic_codemixed/[Language]/train_synthetic.csv` etc.

---

## 📁 Data on Disk (Already Exists)

```
artifacts/data/
├── ilsum_processed/
│   ├── Hindi/       train.csv (~128MB), val.csv (~27MB), test.csv (~27MB)
│   ├── Bengali/     train/val/test.csv
│   ├── English/     train/val/test.csv
│   └── Gujarati/    train/val/test.csv
├── preprocessed/
│   ├── Hindi/       train_clean.csv (~128MB), val_clean.csv, test_clean.csv
│   ├── Bengali/     train_clean/val_clean/test_clean.csv
│   ├── English/     train_clean/val_clean/test_clean.csv
│   ├── Gujarati/    train_clean/val_clean/test_clean.csv
│   └── real_codemixed/ train_clean/val_clean/test_clean.csv
├── synthetic_codemixed/
│   ├── Hindi/       train_synthetic.csv (~127MB), val_synthetic.csv, test_synthetic.csv
│   ├── Bengali/     train_synthetic/val_synthetic/test_synthetic.csv
│   └── Gujarati/    train_synthetic/val_synthetic/test_synthetic.csv
├── real_codemixed/
│   └──              train.csv (~484KB), val.csv (~105KB), test.csv (~109KB)
└── dictionaries/
    ├── hindi_english.json
    ├── bengali_english.json
    └── gujarati_english.json
```

---

## 🏋️ Training Strategy (3 Phases)

### Phase 1: Monolingual Training
- Dataset: ILSUM preprocessed (all 4 languages combined or per-language)
- Goal: Teach model core summarization ability
- Config: `src/configs/phase1_train.yaml`
  - epochs: 5, lr: 3e-5, batch: 2, grad_accum: 8, mixed_precision: true
- Checkpoint saved to: `artifacts/checkpoints/phase1/`
- Status: **NOT STARTED** (checkpoint dir is empty)

### Phase 2: Code-Mixed Adaptation
- Dataset: 70% synthetic code-mixed + 30% real HinGE
- Loads FROM: `artifacts/checkpoints/phase1/`
- Config: `src/configs/phase2_train.yaml`
  - epochs: 5, lr: 3e-5, batch: 2, grad_accum: 8
- Checkpoint: `artifacts/checkpoints/phase2/`
- Status: **NOT STARTED**

### Phase 3: Stabilization (Optional but Recommended)
- Dataset: Phase 2 data + 20% ILSUM mix (prevents forgetting)
- Loads FROM: `artifacts/checkpoints/phase2/`
- Config: `src/configs/phase3_train.yaml`
  - epochs: 3, lr: 1.5e-5, batch: 2, grad_accum: 8
- Checkpoint: `artifacts/checkpoints/phase3/`
- Status: **NOT STARTED**

---

## ⚙️ Inference Config (`src/configs/inference_config.yaml`)
```yaml
num_beams: 4
length_penalty: 1.0
no_repeat_ngram_size: 3
max_new_tokens: 128
```

---

## 📁 Full File Status

### ✅ FULLY IMPLEMENTED
| File | Lines | Notes |
|------|-------|-------|
| `src/components/data/dataset_loader.py` | 366 | `DatasetLoader` class, `initiate_data_loading()` |
| `src/components/data/preprocessor.py` | 490 | `TextPreprocessor` class, `initiate_preprocessing()` |
| `src/components/data/synthetic_generator.py` | 784 | `SyntheticDataGenerator` class, `initiate_synthetic_generation()` |
| `src/exception/exception.py` | 22 | `CodeMixedSummarizationException(error, sys)` |
| `src/logging/logger.py` | 16 | Timestamped file logging to `src/logs/` |
| `setup.py` | 43 | Package `codemix_summarizer`, CLI entry point |
| All YAML configs | — | model_config, phase1/2/3_train, inference_config |

### ❌ STUB ONLY (docstring, no implementation)
| File | What It Should Do |
|------|-------------------|
| `src/components/model/model_loader.py` | Load `ai4bharat/IndicBART` via `AutoModelForSeq2SeqLM`, CPU/GPU placement |
| `src/components/model/tokenizer_utils.py` | Add special lang tag tokens, tokenize inputs (512 max), decode outputs |
| `src/components/model/trainer.py` | `Seq2SeqTrainer` wrapper, reads phase YAML config |
| `src/components/evaluation/rouge_scorer.py` | ROUGE-1, ROUGE-2, ROUGE-L via `rouge-score` |
| `src/components/evaluation/bert_scorer.py` | BERTScore semantic similarity (optional) |
| `src/components/inference/predictor.py` | Full inference: preprocess → tag → tokenize → generate → decode |
| `src/components/inference/language_detector.py` | Auto-detect dominant language for tag selection |
| `src/pipelines/train_phase1.py` | Run Phase 1 end-to-end |
| `src/pipelines/train_phase2.py` | Run Phase 2 end-to-end |
| `src/pipelines/train_phase3.py` | Run Phase 3 end-to-end |
| `src/pipelines/generate_synthetic.py` | Standalone synthetic data generation script |
| `src/pipelines/evaluate.py` | Evaluation pipeline |
| `src/pipelines/inference_pipeline.py` | CLI inference script |
| `app.py` | FastAPI or Gradio web demo |
| `tests/test_preprocessor.py` | Unit tests for preprocessor |
| `tests/test_synthetic_gen.py` | Unit tests for synthetic generator |
| `tests/test_predictor.py` | Integration test for inference |

---

## 🔨 Implementation Order (Dependency Chain)

The next files to implement are (in this order):

```
Phase 1 Training Block:
1. model_loader.py       ← Load IndicBART + tokenizer, device placement
2. tokenizer_utils.py    ← Tokenize datasets, add special tokens
3. trainer.py            ← Seq2SeqTrainer wrapper, reads YAML config
4. train_phase1.py       ← Orchestrates Phase 1 end-to-end

Then Evaluation:
5. rouge_scorer.py       ← ROUGE scores after each phase

Then Phase 2 & 3:
6. train_phase2.py
7. train_phase3.py

Then Inference:
8. predictor.py
9. language_detector.py
10. inference_pipeline.py

Then App:
11. app.py (FastAPI or Gradio)

Finally Tests:
12. test_preprocessor.py, test_synthetic_gen.py, test_predictor.py
```

---

## ⚠️ Known Issues / Things to Fix

1. **`model_config.yaml` language tag mismatch:**  
   Config lists `<2ta>` (Tamil) but the project uses `<2bn>` (Bengali) and `<2gu>` (Gujarati).  
   Tamil is not used anywhere. Fix the config to list `<2hi>`, `<2bn>`, `<2gu>`, `<2en>`.

2. **Missing dependencies in `requirements.txt`:**  
   `scikit-learn` and `pandas` are used but not listed. Add them.

3. **Romanized Hinglish handling:**  
   Text like `"aur bhai kya kar rha"` (Hindi in Latin script) is NOT Devanagari.  
   The synthetic generator's `_is_latin_script()` check skips such words.  
   HinGE dataset naturally contains Romanized Hinglish — the model will learn it via Phase 2.  
   The optional `transliterator.py` (stub, 161 bytes) can later convert Roman→Devanagari using `indic-transliteration`.  
   For now, `language_detector.py` must handle the case where input is all-Latin but is still Hindi (check for common Hindi words like `"aur"`, `"hai"`, `"kya"`, `"nahi"`).

---

## 🛠️ Tech Stack

```
Python >= 3.8
transformers       # HuggingFace — IndicBART model + Seq2SeqTrainer
torch              # PyTorch backend
datasets           # HuggingFace datasets (ILSUM-2.0, HinGE)
sentencepiece      # IndicBART tokenizer dependency
scikit-learn       # train_test_split
pandas             # DataFrames throughout
rouge-score        # ROUGE evaluation
bert-score         # BERTScore evaluation (optional)
fastapi            # Web app backend
uvicorn            # ASGI server for FastAPI
indic-transliteration  # Optional: Romanized → native script
pyyaml             # Read YAML configs
python-dotenv      # .env file support
```

---

## 📐 Code Conventions (Follow These)

1. **Exception pattern** — always wrap in try/except:
   ```python
   except Exception as e:
       raise CodeMixedSummarizationException(e, sys)
   ```

2. **Logging** — import and use like this:
   ```python
   from src.logging.logger import logging
   logging.info("message here")
   ```

3. **Exceptions** — import like this:
   ```python
   from src.exception.exception import CodeMixedSummarizationException
   import sys
   ```

4. **Class structure** — each component is a class with an `initiate_*()` method as the main pipeline entry point (e.g., `initiate_data_loading()`, `initiate_preprocessing()`).

5. **Artifact paths** — always relative to project root:
   - Data: `artifacts/data/...`
   - Checkpoints: `artifacts/checkpoints/phase1/`, `phase2/`, `phase3/`
   - Results: `artifacts/results/`

6. **`if __name__ == "__main__"` blocks** — every component file has a standalone runner for testing individually.
