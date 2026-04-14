"""
Computes BERTScore (precision, recall, F1) between generated and reference
summaries using the `bert-score` library.
"""

import sys
from bert_score import score as bert_score

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


def compute_bertscore(predictions, references):
    """
    Computes average BERTScore across all prediction-reference pairs.
    Returns dict with precision, recall, f1.
    """
    try:
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)

        results = {
            "precision": round(P.mean().item(), 4),
            "recall":    round(R.mean().item(), 4),
            "f1":        round(F1.mean().item(), 4),
        }
        logging.info(f"BERTScore: {results}")
        return results

    except Exception as e:
        raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    preds = ["India won the cricket match today"]
    refs  = ["India defeated Australia in cricket"]
    print(compute_bertscore(preds, refs))
