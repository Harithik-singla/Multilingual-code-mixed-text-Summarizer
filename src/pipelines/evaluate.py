"""
Evaluation pipeline: runs inference on test sets, computes ROUGE + BERTScore,
saves results to artifacts/results/eval_results.json.

Usage: python -m src.pipelines.evaluate
"""

import sys
import os
import json
import pandas as pd

from src.components.inference.predictor import Predictor
from src.components.evaluation.rouge_scorer import compute_rouge
from src.components.evaluation.bert_scorer import compute_bertscore
from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


DATA_DIR    = "artifacts/data"
RESULTS_DIR = "artifacts/results"
MAX_SAMPLES = 500   # limit per split to keep eval fast


TEST_FILES = [
    ("HinGE",    os.path.join(DATA_DIR, "preprocessed/real_codemixed/test_clean.csv")),
    ("Hindi",    os.path.join(DATA_DIR, "preprocessed/Hindi/test_clean.csv")),
    ("Bengali",  os.path.join(DATA_DIR, "preprocessed/Bengali/test_clean.csv")),
    ("Gujarati", os.path.join(DATA_DIR, "preprocessed/Gujarati/test_clean.csv")),
]


def evaluate_split(predictor, csv_path, split_name):
    df = pd.read_csv(csv_path).dropna(subset=["text", "summary"])
    df = df.head(MAX_SAMPLES)

    logging.info(f"Evaluating [{split_name}] — {len(df)} samples")

    predictions = [predictor.predict(text) for text in df["text"]]
    references  = df["summary"].tolist()

    rouge  = compute_rouge(predictions, references)
    bscore = compute_bertscore(predictions, references)

    return {"rouge": rouge, "bertscore": bscore, "samples": len(df)}


def main():
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        predictor = Predictor()
        all_results = {}

        for split_name, csv_path in TEST_FILES:
            if not os.path.exists(csv_path):
                logging.warning(f"Skipping {split_name} — file not found: {csv_path}")
                continue
            all_results[split_name] = evaluate_split(predictor, csv_path, split_name)

        output_path = os.path.join(RESULTS_DIR, "eval_results.json")
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logging.info(f"Results saved → {output_path}")

        print("\n===== Evaluation Results =====")
        for split, res in all_results.items():
            print(f"\n[{split}] ({res['samples']} samples)")
            print(f"  ROUGE-1: {res['rouge']['rouge1']}  ROUGE-2: {res['rouge']['rouge2']}  ROUGE-L: {res['rouge']['rougeL']}")
            print(f"  BERTScore F1: {res['bertscore']['f1']}")

    except Exception as e:
        raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    main()
