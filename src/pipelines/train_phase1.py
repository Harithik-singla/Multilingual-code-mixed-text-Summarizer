"""
Phase 1 Training Script.

Loads preprocessed ILSUM data for all 4 languages, combines them,
tokenizes, and trains IndicBART (Phase 1 - monolingual summarization).

Run from project root:
    python -m src.pipelines.train_phase1
"""

import sys
import os
import pandas as pd

from src.components.model.model_loader import ModelLoader
from src.components.model.tokenizer_utils import TokenizerUtils
from src.components.model.trainer import ModelTrainer
from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


# ── Settings ──────────────────────────────────────────────────────────
PREPROCESSED_DIR  = "artifacts/data/preprocessed"
MODEL_CONFIG      = "src/configs/model_config.yaml"
PHASE1_CONFIG     = "src/configs/phase1_train.yaml"
LANGUAGES         = ["Hindi", "Bengali", "English", "Gujarati"]

# Set to a small number (e.g. 200) for a quick test run
# Set to None to use ALL data for full training
MAX_SAMPLES_PER_LANGUAGE = 200  # change to None for full training
# ──────────────────────────────────────────────────────────────────────


def load_combined_split(split):
    """
    Loads train/val/test CSVs for all 4 languages and combines them.
    split: 'train', 'val', or 'test'
    """
    all_dfs = []

    for lang in LANGUAGES:
        csv_path = os.path.join(PREPROCESSED_DIR, lang, f"{split}_clean.csv")

        if not os.path.exists(csv_path):
            logging.warning(f"File not found, skipping: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Optional: limit rows per language for quick tests
        if MAX_SAMPLES_PER_LANGUAGE is not None:
            df = df.head(MAX_SAMPLES_PER_LANGUAGE)

        all_dfs.append(df)
        logging.info(f"Loaded [{lang}/{split}]: {len(df)} rows")

    # Combine all languages and shuffle so batches have mixed languages
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info(f"Combined [{split}]: {len(combined)} total rows")
    return combined


def main():
    try:
        print("=" * 50)
        print("PHASE 1 TRAINING STARTED")
        print("=" * 50)

        # Step 1: Load data
        print("\nStep 1: Loading ILSUM data...")
        train_df = load_combined_split("train")
        val_df   = load_combined_split("val")
        print(f"  Train rows: {len(train_df)}")
        print(f"  Val rows  : {len(val_df)}")

        # Step 2: Load model and tokenizer
        print("\nStep 2: Loading IndicBART model...")
        loader = ModelLoader(config_path=MODEL_CONFIG)
        model, tokenizer, device = loader.load()
        print(f"  Device: {device}")

        # Step 3: Set up tokenizer (adds language tag tokens)
        print("\nStep 3: Setting up tokenizer...")
        tok = TokenizerUtils(
            tokenizer=tokenizer,
            max_input_length=loader.max_input_length,
            max_output_length=loader.max_output_length
        )
        print(f"  Vocab size: {tok.vocab_size()}")

        # Step 4: Tokenize datasets
        print("\nStep 4: Tokenizing datasets (this may take a few minutes)...")
        train_dataset = tok.tokenize_dataframe(train_df)
        val_dataset   = tok.tokenize_dataframe(val_df)
        print(f"  Train examples: {len(train_dataset)}")
        print(f"  Val examples  : {len(val_dataset)}")

        # Step 5: Train
        print("\nStep 5: Starting training...")
        trainer = ModelTrainer(
            phase=1,
            config_path=PHASE1_CONFIG,
            model=model,
            tokenizer=tokenizer,
            vocab_size=tok.vocab_size()
        )

        train_loss = trainer.train(train_dataset, val_dataset)

        print("\n" + "=" * 50)
        print("PHASE 1 TRAINING COMPLETE!")
        print(f"  Train Loss : {train_loss}")
        print(f"  Checkpoint : artifacts/checkpoints/phase1/")
        print("=" * 50)
        print("\nNext: run Phase 2 training")

    except Exception as e:
        raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    main()
