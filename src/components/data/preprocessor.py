"""
Cleans text data and adds IndicBART language tags.

Takes raw CSVs from dataset_loader, cleans the text (removes URLs,
HTML, noisy characters), adds language tags like <2hi>, and saves
cleaned CSVs.

Run after dataset_loader:
    python -m src.components.data.preprocessor
"""

import sys
import os
import re
import unicodedata
import pandas as pd

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


# IndicBART language tag for each language
LANGUAGE_TAG = {
    "Hindi":    "<2hi>",
    "Bengali":  "<2bn>",
    "English":  "<2en>",
    "Gujarati": "<2gu>",
    "Hinglish": "<2hi>",   # Hindi is dominant in Hinglish
}

ILSUM_LANGUAGES = ["Hindi", "Bengali", "English", "Gujarati"]
DATA_DIR        = "artifacts/data"


def clean_text(text):
    """
    Cleans a single text string:
    1. NFC unicode normalization (fixes Indic script encoding issues)
    2. Remove zero-width characters (common in Indic text)
    3. Remove URLs
    4. Remove HTML tags and entities
    5. Remove control characters
    6. Normalize whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Zero-width characters (invisible chars that cause tokenizer issues)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)

    # 3. URLs
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

    # 4. HTML tags and entities
    text = re.sub(r"&[a-zA-Z]+;|&#[0-9]+;", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # 5. Control characters (keep normal spaces/newlines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # 6. Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df, language):
    """
    Cleans the text and summary columns, adds language tag to text.
    Returns cleaned DataFrame.
    """
    df = df.copy()

    # Clean both columns
    df["text"]    = df["text"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)

    # Add language tag to the START of text (IndicBART needs this)
    # e.g. "<2hi> आज मौसम अच्छा है"
    tag = LANGUAGE_TAG[language]
    df["text"] = tag + " " + df["text"]

    # Drop rows that became empty after cleaning
    df = df[df["text"].str.strip().astype(bool)]
    df = df[df["summary"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    return df


class TextPreprocessor:
    def __init__(self):
        try:
            self.output_dir = os.path.join(DATA_DIR, "preprocessed")
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"TextPreprocessor ready — output: {self.output_dir}")
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def preprocess_split(self, input_path, output_path, language):
        """Reads a CSV, cleans it, saves the cleaned version."""
        try:
            df = pd.read_csv(input_path)
            logging.info(f"Preprocessing [{language}] {input_path} — {len(df)} rows")

            cleaned_df = preprocess_dataframe(df, language)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cleaned_df.to_csv(output_path, index=False)
            logging.info(f"  Saved {len(cleaned_df)} rows → {output_path}")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def run(self):
        """
        Preprocesses all ILSUM languages + HinGE real code-mixed data.
        Reads from ilsum_processed/ and real_codemixed/, saves to preprocessed/.
        """
        try:
            logging.info("Starting preprocessing pipeline...")

            # Preprocess each ILSUM language
            for lang in ILSUM_LANGUAGES:
                for split in ["train", "val", "test"]:
                    input_path  = os.path.join(DATA_DIR, "ilsum_processed", lang, f"{split}.csv")
                    output_path = os.path.join(self.output_dir, lang, f"{split}_clean.csv")

                    if not os.path.exists(input_path):
                        logging.warning(f"Not found, skipping: {input_path}")
                        continue

                    self.preprocess_split(input_path, output_path, lang)

            # Preprocess HinGE real code-mixed data
            for split in ["train", "val", "test"]:
                input_path  = os.path.join(DATA_DIR, "real_codemixed", f"{split}.csv")
                output_path = os.path.join(self.output_dir, "real_codemixed", f"{split}_clean.csv")

                if not os.path.exists(input_path):
                    logging.warning(f"Not found, skipping: {input_path}")
                    continue

                self.preprocess_split(input_path, output_path, "Hinglish")

            logging.info("Preprocessing pipeline complete!")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.run()
    print("Preprocessing complete! Cleaned files saved to artifacts/data/preprocessed/")