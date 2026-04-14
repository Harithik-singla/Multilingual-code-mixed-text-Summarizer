"""
Computes ROUGE-1, ROUGE-2, and ROUGE-L scores on model predictions vs reference
summaries using the `rouge-score` library.
"""

import sys
from rouge_score import rouge_scorer as rs

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


def compute_rouge(predictions, references):
    """
    Computes average ROUGE scores across all prediction-reference pairs.
    Returns dict with rouge1, rouge2, rougeL (each has precision, recall, fmeasure).
    """
    try:
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        n = len(predictions)

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in totals:
                totals[key] += scores[key].fmeasure

        results = {key: round(totals[key] / n, 4) for key in totals}
        logging.info(f"ROUGE scores: {results}")
        return results

    except Exception as e:
        raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    preds = ["India won the cricket match today"]
    refs  = ["India defeated Australia in cricket"]
    print(compute_rouge(preds, refs))
