"""
Phase 3 Training: Stabilization.
Loads Phase 2 checkpoint, trains on Phase 2 data mix + 20% ILSUM.

Run from project root:
    python -m src.pipelines.train_phase3
"""

import sys
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.components.model.model_loader import ModelLoader
from src.components.model.tokenizer_utils import TokenizerUtils
from src.components.model.trainer import ModelTrainer
from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


SYNTHETIC_DIR   = "artifacts/data/synthetic_codemixed"
REAL_DIR        = "artifacts/data/preprocessed/real_codemixed"
ILSUM_DIR       = "artifacts/data/preprocessed"
PHASE2_CKPT     = "artifacts/checkpoints/phase2"
PHASE3_CONFIG   = "src/configs/phase3_train.yaml"
SYNTHETIC_LANGS = ["Hindi", "Bengali", "Gujarati"]
ILSUM_LANGS     = ["Hindi", "Bengali", "English", "Gujarati"]

MAX_SAMPLES_PER_SYNTHETIC = None
MAX_SAMPLES_REAL          = None
MAX_SAMPLES_PER_ILSUM     = None


def load_split(split):
    all_dfs = []

    # Synthetic data
    for lang in SYNTHETIC_LANGS:
        path = os.path.join(SYNTHETIC_DIR, lang, f"{split}_synthetic.csv")
        if not os.path.exists(path):
            logging.warning(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        if MAX_SAMPLES_PER_SYNTHETIC:
            df = df.head(MAX_SAMPLES_PER_SYNTHETIC)
        all_dfs.append(df)

    synthetic_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Real HinGE data
    real_path = os.path.join(REAL_DIR, f"{split}_clean.csv")
    real_df = pd.DataFrame()
    if os.path.exists(real_path):
        real_df = pd.read_csv(real_path)
        if MAX_SAMPLES_REAL:
            real_df = real_df.head(MAX_SAMPLES_REAL)

    # 70/30 synthetic/real mix (same as phase 2)
    n_synthetic = int(len(synthetic_df) * 0.7)
    n_real      = int(len(real_df) * 0.3)
    phase2_mix  = pd.concat([
        synthetic_df.sample(n=min(n_synthetic, len(synthetic_df)), random_state=42),
        real_df.sample(n=min(n_real, len(real_df)), random_state=42)
    ], ignore_index=True)

    # 20% ILSUM mix
    ilsum_dfs = []
    for lang in ILSUM_LANGS:
        path = os.path.join(ILSUM_DIR, lang, f"{split}_clean.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if MAX_SAMPLES_PER_ILSUM:
            df = df.head(MAX_SAMPLES_PER_ILSUM)
        ilsum_dfs.append(df)

    ilsum_df = pd.concat(ilsum_dfs, ignore_index=True) if ilsum_dfs else pd.DataFrame()
    n_ilsum  = int(len(phase2_mix) * 0.2)
    ilsum_sample = ilsum_df.sample(n=min(n_ilsum, len(ilsum_df)), random_state=42)

    combined = pd.concat([phase2_mix, ilsum_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(f"Phase 3 [{split}]: {len(combined)} rows")
    return combined


def main():
    try:
        train_df = load_split("train")
        val_df   = load_split("val")

        loader = ModelLoader(config_path="src/configs/model_config.yaml")
        device = loader.get_device()

        tokenizer = AutoTokenizer.from_pretrained(PHASE2_CKPT, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(PHASE2_CKPT).to(device)
        logging.info(f"Loaded Phase 2 checkpoint from {PHASE2_CKPT}, device: {device}")

        tok = TokenizerUtils(
            tokenizer=tokenizer,
            max_input_length=loader.max_input_length,
            max_output_length=loader.max_output_length
        )

        train_dataset = tok.tokenize_dataframe(train_df)
        val_dataset   = tok.tokenize_dataframe(val_df)

        trainer = ModelTrainer(
            phase=3,
            config_path=PHASE3_CONFIG,
            model=model,
            tokenizer=tokenizer,
            vocab_size=tok.vocab_size()
        )

        train_loss = trainer.train(train_dataset, val_dataset)
        logging.info(f"Phase 3 done — loss: {train_loss}")

    except Exception as e:
        raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    main()
