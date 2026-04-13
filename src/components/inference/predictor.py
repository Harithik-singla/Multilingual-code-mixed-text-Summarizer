"""
End-to-end inference: detects language, tokenizes input, generates summary.
Loads the best available checkpoint (phase3 → phase2 → phase1 fallback).
"""

import sys
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.components.inference.language_detector import detect_language_tag
from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


CHECKPOINT_PRIORITY = [
    "artifacts/checkpoints/phase3",
    "artifacts/checkpoints/phase2",
    "artifacts/checkpoints/phase1",
]
INFERENCE_CONFIG = "src/configs/inference_config.yaml"


def _load_checkpoint():
    for ckpt in CHECKPOINT_PRIORITY:
        if os.path.exists(os.path.join(ckpt, "config.json")):
            logging.info(f"Loading checkpoint: {ckpt}")
            return ckpt
    raise FileNotFoundError("No trained checkpoint found. Run training first.")


class Predictor:
    def __init__(self):
        try:
            with open(INFERENCE_CONFIG, "r") as f:
                self.config = yaml.safe_load(f)

            ckpt = _load_checkpoint()
            self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
            self.model     = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(self.device)
            self.model.eval()

            logging.info(f"Predictor ready on {self.device}")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def predict(self, text):
        try:
            lang_tag   = detect_language_tag(text)
            tagged_text = f"{lang_tag} {text.strip()}"

            inputs = self.tokenizer(
                tagged_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    num_beams=self.config.get("num_beams", 4),
                    length_penalty=self.config.get("length_penalty", 1.0),
                    no_repeat_ngram_size=self.config.get("no_repeat_ngram_size", 3),
                    max_new_tokens=self.config.get("max_new_tokens", 128),
                )

            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    predictor = Predictor()
    test_text = "aaj bharat ne cricket match jeeta aur poori team ne bahut achha khela"
    print(f"Input : {test_text}")
    print(f"Summary: {predictor.predict(test_text)}")
