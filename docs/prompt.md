You are helping set up a machine learning project called "Multilingual Code-Mixed Text Summarization".

## What We Are Building

We are building a deep learning system that can summarize code-mixed text — text that mixes Indian languages (Hindi, Tamil, etc.) with English (e.g., Hinglish like "aaj meeting hai and we need to prepare report"). This is called code-mixed or code-switched text and is extremely common in real-world Indian social media, news comments, and messaging.

The model we use is IndicBART (ai4bharat/IndicBART), a multilingual transformer (~244M parameters) pretrained on Indian languages. We fine-tune it in 3 phases:
- Phase 1: Train on monolingual Indian language summarization data (ILSUM dataset) to learn summarization
- Phase 2: Fine-tune on code-mixed data — 70% synthetically generated + 30% real (Hinglish/GupShup dataset)
- Phase 3 (optional): Mix in a small amount of ILSUM again to prevent the model from forgetting how to summarize cleanly

The synthetic data generation works by taking monolingual sentences and randomly replacing 20–40% of content words (nouns, adjectives) with their English equivalents, producing realistic code-mixed training examples without needing a large real dataset.

The inference pipeline works as:
Input (code-mixed text) → Basic preprocessing → Optional transliteration → Add language tag (<2hi>, <2ta>, <2en>) → Tokenize → IndicBART generates summary → Decode output

We evaluate using ROUGE-1, ROUGE-2, ROUGE-L scores, and optionally BERTScore.

---

## Your Task

Create the following complete project folder and file structure on disk. For every file, create it as an EMPTY file with just a top-of-file comment explaining what that file is responsible for. Do NOT write any actual implementation code — just the file with a single docstring or comment block at the top describing its purpose.

Use Python triple-quoted docstrings for .py files, # comments for .yaml and .txt files, and standard markdown for .md files.

---

## Project Structure to Create
codemix-summarizer/
├── setup.py
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── .gitignore
│
├── configs/
│   ├── model_config.yaml
│   ├── phase1_train.yaml
│   ├── phase2_train.yaml
│   ├── phase3_train.yaml
│   └── inference_config.yaml
│
├── components/
│   ├── init.py
│   ├── data/
│   │   ├── init.py
│   │   ├── dataset_loader.py
│   │   ├── preprocessor.py
│   │   ├── transliterator.py
│   │   └── synthetic_generator.py
│   ├── model/
│   │   ├── init.py
│   │   ├── model_loader.py
│   │   ├── trainer.py
│   │   └── tokenizer_utils.py
│   ├── evaluation/
│   │   ├── init.py
│   │   ├── rouge_scorer.py
│   │   └── bert_scorer.py
│   └── inference/
│       ├── init.py
│       ├── predictor.py
│       └── language_detector.py
│
├── pipelines/
│   ├── train_phase1.py
│   ├── train_phase2.py
│   ├── train_phase3.py
│   ├── evaluate.py
│   ├── generate_synthetic.py
│   └── inference_pipeline.py
│
├── artifacts/
│   ├── data/
│   │   ├── ilsum_processed/
│   │   ├── synthetic_codemixed/
│   │   └── real_codemixed/
│   ├── checkpoints/
│   │   ├── phase1/
│   │   ├── phase2/
│   │   └── phase3/
│   └── results/
│       ├── baseline_rouge.json
│       ├── final_rouge.json
│       └── sample_outputs.txt
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_synthetic_data_demo.ipynb
│   └── 03_training_curves.ipynb
│
└── tests/
├── test_preprocessor.py
├── test_synthetic_gen.py
└── test_predictor.py

---

## Comment/Docstring Content Per File

For each file, include a comment that describes exactly what that file does in the context of this project. Use the following descriptions:

**setup.py** — Package installer. Defines the `codemix_summarizer` package and CLI entry points for `train` and `summarize` commands.

**app.py** — FastAPI or Gradio web app. Loads the trained IndicBART model and exposes a /summarize endpoint plus a simple demo UI for testing.

**requirements.txt** — All pip dependencies: transformers, torch, datasets, sentencepiece, rouge-score, bert-score, fastapi, uvicorn, indic-transliteration, pyyaml, python-dotenv.

**.env.example** — Template for environment variables: MODEL_PATH, HF_TOKEN, LOG_LEVEL, DEVICE.

**README.md** — Project overview, setup instructions, how to run each training phase, how to launch the app.

**.gitignore** — Ignores venv/, artifacts/checkpoints/, artifacts/data/, __pycache__, *.pyc, .env

**configs/model_config.yaml** — Model name (ai4bharat/IndicBART), max_input_length (512-768), max_output_length, supported language tags (<2hi>, <2ta>, <2en>).

**configs/phase1_train.yaml** — Phase 1 hyperparameters: dataset=ILSUM, epochs=3-5, lr=3e-5, batch_size=2-4, gradient_accumulation_steps=8, mixed_precision=true.

**configs/phase2_train.yaml** — Phase 2 hyperparameters: synthetic_ratio=0.7, real_ratio=0.3, load_from=artifacts/checkpoints/phase1, same lr and batch settings.

**configs/phase3_train.yaml** — Phase 3 hyperparameters: ilsum_mix_ratio=0.2, load_from=artifacts/checkpoints/phase2, lower lr for stability.

**configs/inference_config.yaml** — Generation settings: num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, max_new_tokens=128.

**components/data/dataset_loader.py** — Loads ILSUM dataset and real code-mixed datasets (Hinglish/GupShup) using HuggingFace `datasets`. Returns train/val/test splits.

**components/data/preprocessor.py** — Cleans input text: removes extra spaces, strips noisy characters, normalizes unicode, and prepends the appropriate language tag (<2hi>, <2ta>, <2en>).

**components/data/transliterator.py** — Optional transliteration step using the `indic-transliteration` library. Converts Romanized Indic words to native script (e.g., "aaj" → "आज").

**components/data/synthetic_generator.py** — Core data augmentation module. Takes monolingual Hindi/Tamil sentences and randomly replaces 20–40% of content words (nouns, adjectives) with English equivalents to generate synthetic code-mixed training data.

**components/model/model_loader.py** — Loads ai4bharat/IndicBART using HuggingFace AutoModelForSeq2SeqLM and its tokenizer. Handles device placement (CPU/GPU).

**components/model/trainer.py** — Wraps HuggingFace Trainer for seq2seq training. Accepts a phase config YAML and runs the appropriate training phase on the given dataset.

**components/model/tokenizer_utils.py** — Tokenizer helper functions: adds special language tag tokens, handles input truncation to max_input_length, and decodes generated token ids back to text.

**components/evaluation/rouge_scorer.py** — Computes ROUGE-1, ROUGE-2, and ROUGE-L scores on model predictions vs reference summaries using the `rouge-score` library.

**components/evaluation/bert_scorer.py** — Optional semantic evaluation using BERTScore. Measures similarity between generated and reference summaries at the embedding level.

**components/inference/predictor.py** — End-to-end inference: takes raw code-mixed input text, runs preprocessing, language tagging, tokenization, calls the model, and returns the decoded summary.

**components/inference/language_detector.py** — Detects the dominant language of input text to automatically select the correct IndicBART language tag without manual input.

**pipelines/train_phase1.py** — Runnable script: loads ILSUM dataset, runs preprocessor, initializes IndicBART, runs Phase 1 training using phase1_train.yaml config, saves checkpoint to artifacts/checkpoints/phase1/.

**pipelines/train_phase2.py** — Runnable script: loads synthetic + real code-mixed data at 70/30 ratio, loads Phase 1 checkpoint, runs Phase 2 fine-tuning using phase2_train.yaml, saves to artifacts/checkpoints/phase2/.

**pipelines/train_phase3.py** — Runnable script: loads Phase 2 checkpoint, mixes in small ILSUM portion, runs stabilization fine-tuning using phase3_train.yaml, saves final model to artifacts/checkpoints/phase3/.

**pipelines/evaluate.py** — Loads a trained checkpoint, runs inference on the test set, computes and saves ROUGE + BERTScore to artifacts/results/.

**pipelines/generate_synthetic.py** — Standalone script: reads ILSUM training data, applies synthetic_generator to produce code-mixed examples, saves the output dataset to artifacts/data/synthetic_codemixed/.

**pipelines/inference_pipeline.py** — CLI script: accepts input text from command line or stdin, runs the full predictor pipeline, and prints the generated summary.

**tests/test_preprocessor.py** — Unit tests for preprocessor.py: verifies noise removal, language tag injection, unicode normalization, and edge cases like empty strings.

**tests/test_synthetic_gen.py** — Unit tests for synthetic_generator.py: verifies that 20–40% of words are replaced, sentence structure is preserved, and only content words are substituted.

**tests/test_predictor.py** — End-to-end integration test for predictor.py: loads a tiny/dummy checkpoint and verifies the full inference pipeline runs without errors.

---

## Additional Instructions

1. For all `artifacts/data/ilsum_processed/`, `artifacts/data/synthetic_codemixed/`, `artifacts/data/real_codemixed/`, `artifacts/checkpoints/phase1/`, `artifacts/checkpoints/phase2/`, `artifacts/checkpoints/phase3/` — these are empty directories. Create a `.gitkeep` file inside each to preserve them in git.

2. For the 3 notebook files (`.ipynb`), create valid minimal Jupyter notebooks with just a single markdown cell describing the notebook's purpose — no code cells needed.

3. The `.gitignore` should actually contain the ignore rules (not just a comment), since it is a config file not a code file.

4. For `requirements.txt`, list the actual package names (one per line) based on the descriptions above.

5. After creating all files, print a confirmation tree of the full structure using the `tree` command or equivalent so I can verify everything was created correctly.