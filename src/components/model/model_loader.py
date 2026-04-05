"""
Loads ai4bharat/IndicBART using HuggingFace AutoModelForSeq2SeqLM and its
tokenizer. Handles device placement (CPU/GPU) and reads model_config.yaml.
"""

import sys
import os
import yaml
import torch
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

DEFAULT_MODEL_CONFIG = "src/configs/model_config.yaml"
DEFAULT_CACHE_DIR    = "artifacts/model_cache"   # avoids re-downloading


class ModelLoader:
    """
    Loads and returns the IndicBART model and tokenizer.

    Responsibilities:
        - Read model name and config from model_config.yaml
        - Download (or load from cache) ai4bharat/IndicBART tokenizer + model
        - Detect available device (CUDA GPU > MPS Apple Silicon > CPU)
        - Move model to the correct device
        - Expose the loaded model and tokenizer via load() method

    Usage:
        loader = ModelLoader()
        model, tokenizer, device = loader.load()

    Or with a custom config:
        loader = ModelLoader(config_path="src/configs/model_config.yaml")
        model, tokenizer, device = loader.load()
    """

    def __init__(
        self,
        config_path: str = DEFAULT_MODEL_CONFIG,
        cache_dir: str   = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize ModelLoader.

        Args:
            config_path : Path to model_config.yaml.
                          Defaults to 'src/configs/model_config.yaml'.
            cache_dir   : Directory to cache downloaded HuggingFace model files.
                          Avoids re-downloading on every run.
                          Defaults to 'artifacts/model_cache'.
        """
        try:
            self.config_path = config_path
            self.cache_dir   = cache_dir

            os.makedirs(self.cache_dir, exist_ok=True)

            # Load model config YAML
            self.config = self._load_config()

            self.model_name       = self.config.get("model_name", "ai4bharat/IndicBART")
            self.max_input_length = self.config.get("max_input_length", 512)
            self.max_output_length = self.config.get("max_output_length", 128)
            self.language_tags    = self.config.get("language_tags", [
                "<2hi>", "<2bn>", "<2gu>", "<2en>"
            ])

            logging.info(
                f"ModelLoader initialized\n"
                f"  model_name       : {self.model_name}\n"
                f"  config_path      : {self.config_path}\n"
                f"  cache_dir        : {self.cache_dir}\n"
                f"  max_input_length : {self.max_input_length}\n"
                f"  max_output_length: {self.max_output_length}\n"
                f"  language_tags    : {self.language_tags}"
            )

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_config(self) -> dict:
        """
        Load and return the model_config.yaml as a dictionary.

        Returns:
            Dict with keys: model_name, max_input_length,
                            max_output_length, language_tags
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(
                    f"Model config not found at '{self.config_path}'. "
                    f"Expected file: src/configs/model_config.yaml"
                )

            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            logging.info(f"Loaded model config from: {self.config_path}")
            return config

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def _detect_device(self) -> torch.device:
        """
        Detect the best available compute device.

        Priority:
            1. CUDA  — NVIDIA GPU (fastest)
            2. MPS   — Apple Silicon GPU (macOS)
            3. CPU   — Fallback (slowest, always available)

        Returns:
            torch.device object
        """
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                logging.info(f"Device: CUDA GPU detected — {gpu_name}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logging.info("Device: Apple MPS (Metal) detected")
            else:
                device = torch.device("cpu")
                logging.info("Device: No GPU found — using CPU (training will be slow)")

            return device

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def _load_tokenizer(self) -> AutoTokenizer:
        """
        Download (or load from cache) the IndicBART tokenizer.

        IndicBART uses a SentencePiece-based tokenizer.
        We use use_fast=False because the fast tokenizer has known
        issues with some Indic scripts in older transformers versions.

        Returns:
            Loaded AutoTokenizer
        """
        try:
            logging.info(f"Loading tokenizer: {self.model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast    = False,     # SentencePiece tokenizer for Indic
                cache_dir   = self.cache_dir,
            )

            logging.info(
                f"Tokenizer loaded — vocab size: {tokenizer.vocab_size}"
            )
            return tokenizer

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def _load_model(self, device: torch.device) -> AutoModelForSeq2SeqLM:
        """
        Download (or load from cache) the IndicBART seq2seq model
        and move it to the specified device.

        Args:
            device: torch.device (cuda / mps / cpu)

        Returns:
            Loaded model on the correct device
        """
        try:
            logging.info(f"Loading model: {self.model_name}...")
            logging.info("  (This may take a few minutes on first run — ~1GB download)")

            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir = self.cache_dir,
            )

            # Count parameters for logging
            total_params     = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            logging.info(
                f"Model loaded\n"
                f"  Total parameters    : {total_params:,}\n"
                f"  Trainable parameters: {trainable_params:,}"
            )

            # Move to device
            model = model.to(device)
            logging.info(f"Model moved to device: {device}")

            return model

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def load(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device]:
        """
        Main entry point. Loads the IndicBART model and tokenizer,
        places the model on the best available device.

        Returns:
            Tuple of (model, tokenizer, device)

            model     : AutoModelForSeq2SeqLM ready for training/inference
            tokenizer : AutoTokenizer for encoding inputs and decoding outputs
            device    : torch.device the model is running on

        Example:
            loader = ModelLoader()
            model, tokenizer, device = loader.load()
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting model loading pipeline...")
            logging.info("=" * 50)

            # Step 1: Detect device
            device    = self._detect_device()

            # Step 2: Load tokenizer
            tokenizer = self._load_tokenizer()

            # Step 3: Load model and move to device
            model     = self._load_model(device)

            logging.info("=" * 50)
            logging.info("Model loading pipeline complete!")
            logging.info(f"  Model    : {self.model_name}")
            logging.info(f"  Device   : {device}")
            logging.info(f"  Vocab    : {tokenizer.vocab_size}")
            logging.info("=" * 50)

            return model, tokenizer, device

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def get_config(self) -> dict:
        """
        Return the loaded model config dictionary.
        Useful for downstream components (tokenizer_utils, trainer)
        to access max_input_length, max_output_length, language_tags.

        Returns:
            Dict with model configuration values
        """
        return self.config


# ------------------------------------------------------------------ #
#  Standalone runner (test this file individually)                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    try:
        print("=" * 50)
        print("Testing ModelLoader...")
        print("=" * 50)

        loader = ModelLoader()
        model, tokenizer, device = loader.load()

        print("\n✅ Model loading successful!")
        print(f"   Model     : {loader.model_name}")
        print(f"   Device    : {device}")
        print(f"   Vocab size: {tokenizer.vocab_size}")

        # Quick sanity check — tokenize a short Hindi sentence
        test_input = "<2hi> आज मौसम अच्छा है"
        tokens     = tokenizer(test_input, return_tensors="pt")
        print(f"\n   Sanity check tokenization:")
        print(f"   Input    : {test_input}")
        print(f"   Token IDs: {tokens['input_ids']}")
        print(f"   Num tokens: {tokens['input_ids'].shape[1]}")

        print("\nModelLoader test complete.")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
