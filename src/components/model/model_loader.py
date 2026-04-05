"""
Loads ai4bharat/IndicBART model and tokenizer.
Handles device placement (CPU/GPU).
"""

import sys
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


class ModelLoader:
    def __init__(self, config_path="src/configs/model_config.yaml"):
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

            self.model_name        = self.config.get("model_name", "ai4bharat/IndicBART")
            self.max_input_length  = self.config.get("max_input_length", 512)
            self.max_output_length = self.config.get("max_output_length", 128)
            self.language_tags     = self.config.get("language_tags", ["<2hi>", "<2bn>", "<2gu>", "<2en>"])

            # Cache downloaded model so it doesn't re-download every run
            self.cache_dir = "artifacts/model_cache"
            os.makedirs(self.cache_dir, exist_ok=True)

            logging.info(f"ModelLoader ready — model: {self.model_name}")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def get_device(self):
        """Returns cuda if GPU available, else cpu."""
        if torch.cuda.is_available():
            logging.info(f"GPU found: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            logging.info("No GPU found, using CPU")
            return torch.device("cpu")

    def load(self):
        """
        Downloads and returns (model, tokenizer, device).
        Call this once at the start of training.
        """
        try:
            device = self.get_device()

            logging.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,       # needed for IndicBART SentencePiece tokenizer
                cache_dir=self.cache_dir
            )

            logging.info("Loading model (may take a few minutes first time)...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            model = model.to(device)

            logging.info(f"Model loaded on {device}")
            return model, tokenizer, device

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    loader = ModelLoader()
    model, tokenizer, device = loader.load()
    print(f"Model loaded! Device: {device}, Vocab size: {tokenizer.vocab_size}")
