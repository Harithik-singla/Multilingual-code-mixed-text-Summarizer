"""
CLI script: accepts input text, runs the predictor, prints the summary.
Usage: python -m src.pipelines.inference_pipeline --text "your text here"
"""

import sys
import argparse

from src.components.inference.predictor import Predictor
from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


def main():
    try:
        parser = argparse.ArgumentParser(description="Code-Mixed Text Summarizer")
        parser.add_argument("--text", type=str, required=True, help="Text to summarize")
        args = parser.parse_args()

        logging.info("Starting inference pipeline...")
        predictor = Predictor()

        summary = predictor.predict(args.text)
        print(f"\nInput  : {args.text}")
        print(f"Summary: {summary}")

    except Exception as e:
        raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    main()
