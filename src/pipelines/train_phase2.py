"""
Phase 2 Training: Code-Mixed Adaptation.
Loads 70% synthetic + 30% real HinGE data, fine-tunes from Phase 1 checkpoint.

Run from project root:
    python -m src.pipelines.train_phase2
"""

import sys
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.components.model.model_loader import ModelLoader
from src.components.model.tokenizer_utils import TokenizerUtils
from src.components.model.trainer import ModelTrainer
from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


SYNTHETIC_DIR  = "artifacts/data/synthetic_codemixed"
REAL_DIR       = "artifacts/data/preprocessed/real_codemixed"
PHASE1_CKPT    = "artifacts/checkpoints/phase1"
PHASE2_CONFIG  = "src/configs/phase2_train.yaml"
SYNTHETIC_LANGS = ["Hindi", "Bengali", "Gujarati"]

MAX_SAMPLES_PER_SYNTHETIC = None  # set to 200 for quick test
MAX_SAMPLES_REAL          = None


def load_split(split):
    all_dfs = []

    # Synthetic data (70% of mix)
    for lang in SYNTHETIC_LANGS:
        path = os.path.join(SYNTHETIC_DIR, lang, f"{split}_synthetic.csv")
        if not os.path.exists(path):
            logging.warning(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        if MAX_SAMPLES_PER_SYNTHETIC:
            df = df.head(MAX_SAMPLES_PER_SYNTHETIC)
        all_dfs.append(df)
        logging.info(f"Synthetic [{lang}/{split}]: {len(df)} rows")

    synthetic_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Real HinGE data (30% of mix)
    real_path = os.path.join(REAL_DIR, f"{split}_clean.csv")
    real_df = pd.DataFrame()
    if os.path.exists(real_path):
        real_df = pd.read_csv(real_path)
        if MAX_SAMPLES_REAL:
            real_df = real_df.head(MAX_SAMPLES_REAL)
        logging.info(f"Real HinGE [{split}]: {len(real_df)} rows")

    # Mix: use all real data, then sample synthetic to make it 70/30 overall
    if len(real_df) == 0 or len(synthetic_df) == 0:
        combined = pd.concat([synthetic_df, real_df], ignore_index=True)
    else:
        # n_real = 30% → n_synthetic should be (70/30) * n_real = 2.33 * n_real
        n_synthetic = min(int(len(real_df) * 7 / 3), len(synthetic_df))
        synthetic_sample = synthetic_df.sample(n=n_synthetic, random_state=42)
        combined = pd.concat([synthetic_sample, real_df], ignore_index=True)

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(f"Combined [{split}]: {len(combined)} rows")
    return combined


def main():
    try:
        train_df = load_split("train")
        val_df   = load_split("val")

        loader = ModelLoader(config_path="src/configs/model_config.yaml")
        device = loader.get_device()

        tokenizer = AutoTokenizer.from_pretrained(PHASE1_CKPT, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(PHASE1_CKPT).to(device)
        logging.info(f"Loaded Phase 1 checkpoint from {PHASE1_CKPT}, device: {device}")

        tok = TokenizerUtils(
            tokenizer=tokenizer,
            max_input_length=loader.max_input_length,
            max_output_length=loader.max_output_length
        )

        train_dataset = tok.tokenize_dataframe(train_df)
        val_dataset   = tok.tokenize_dataframe(val_df)

        trainer = ModelTrainer(
            phase=2,
            config_path=PHASE2_CONFIG,
            model=model,
            tokenizer=tokenizer,
            vocab_size=tok.vocab_size()
        )

        train_loss = trainer.train(train_dataset, val_dataset)
        logging.info(f"Phase 2 done — loss: {train_loss}")

    except Exception as e:
        raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    main()
