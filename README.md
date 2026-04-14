# Multilingual Code-Mixed Text Summarization

A transformer-based abstractive text summarization system that handles **code-mixed multilingual input** (Hinglish, Banglish, Gujarati-English) using **IndicBART**.

## Features

- Summarizes **Hindi, Bengali, Gujarati, English, and code-mixed** text
- Handles **Romanized Hinglish** ("kya kar rha hai") natively
- Transliterates Romanized Bengali/Gujarati to native script automatically
- 3-phase training: Monolingual → Code-Mixed Adaptation → Stabilization
- Streamlit web interface for easy interaction
- ROUGE + BERTScore evaluation pipeline

## Model

- **Base:** `ai4bharat/IndicBART` (~244M params, seq2seq)
- **Datasets:** ILSUM-2.0 (monolingual) + HinGE (Hinglish code-mixed)
- **Training:** 3-phase fine-tuning with bf16 mixed precision

## Project Structure

```
src/
├── components/
│   ├── data/              # Dataset loading, preprocessing, synthetic generation, transliteration
│   ├── model/             # Model loader, tokenizer utilities, trainer
│   ├── inference/         # Language detection, prediction pipeline
│   └── evaluation/        # ROUGE and BERTScore computation
├── pipelines/             # Training (phase 1/2/3), evaluation, CLI inference
├── configs/               # YAML configs for model, training, inference
├── exception/             # Custom exception handler
└── logging/               # Timestamped file logger
app.py                     # Streamlit web demo
tests/                     # Unit and integration tests
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference (CLI)
```bash
python -m src.pipelines.inference_pipeline --text "aaj bharat ne cricket match jeeta"
```

### 3. Launch Web App
```bash
streamlit run app.py
```

### 4. Run Evaluation
```bash
python -m src.pipelines.evaluate
```

### 5. Run Tests
```bash
python -m pytest tests/ -v
```

## Training Phases

| Phase | Data | Purpose |
|-------|------|---------|
| Phase 1 | ILSUM (Hindi + Bengali + English + Gujarati) | Monolingual summarization |
| Phase 2 | 70% synthetic code-mixed + 30% HinGE | Code-mixed adaptation |
| Phase 3 | Phase 2 mix + 20% ILSUM | Stabilization |

## How It Works

1. **Input text** is analyzed to detect the dominant language
2. **Romanized Bengali/Gujarati** is transliterated to native script (Hindi/Hinglish skipped — model handles it natively)
3. **Language tag** (`<2hi>`, `<2bn>`, `<2gu>`) is prepended
4. **IndicBART** generates the summary via beam search
5. **Summary** is returned to the user

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- IndicBART (ai4bharat)
- Streamlit (frontend)
- rouge-score, bert-score (evaluation)
- indic-transliteration (script conversion)

## Author

**Raunaq Mittal** — raunaqmittal2004@gmail.com
