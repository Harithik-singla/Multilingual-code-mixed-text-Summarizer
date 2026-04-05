"""
Generates synthetic code-mixed training data from monolingual ILSUM text.

Takes preprocessed Hindi/Bengali/Gujarati articles and randomly replaces
20-40% of content words with their English equivalents using dictionaries
stored in artifacts/data/dictionaries/.

Example:
    Input : "<2hi> सरकार ने नई योजना शुरू की"
    Output: "<2hi> government ने नई scheme शुरू की"

Run after preprocessor:
    python -m src.components.data.synthetic_generator
"""

import sys
import os
import re
import json
import random
import pandas as pd

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


DATA_DIR   = "artifacts/data"
DICT_DIR   = os.path.join(DATA_DIR, "dictionaries")
OUTPUT_DIR = os.path.join(DATA_DIR, "synthetic_codemixed")

# Languages to process (English skipped — no mixing needed)
SUPPORTED_LANGUAGES = ["Hindi", "Bengali", "Gujarati"]

# Dictionary JSON filenames in artifacts/data/dictionaries/
DICT_FILES = {
    "Hindi":    "hindi_english.json",
    "Bengali":  "bengali_english.json",
    "Gujarati": "gujarati_english.json",
}

# Replacement settings
MIN_REPLACE_RATIO   = 0.20
MAX_REPLACE_RATIO   = 0.40
MIN_SENTENCE_LENGTH = 5     # skip sentences shorter than this


def load_dictionary(lang):
    """
    Loads the word dictionary for a language from JSON file.
    e.g. artifacts/data/dictionaries/hindi_english.json
    """
    json_path = os.path.join(DICT_DIR, DICT_FILES[lang])

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Dictionary not found: {json_path}\n"
            f"Run the data pipeline first to generate dictionary files."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    logging.info(f"Loaded dictionary for {lang}: {len(dictionary)} words from {json_path}")
    return dictionary


def mix_sentence(text, dictionary):
    """
    Takes one text string and randomly replaces 20-40% of content words
    with English equivalents from the dictionary.

    Rules:
    - Skip words shorter than 3 chars (grammar words like है, की, का)
    - Skip words already in Latin script (already English)
    - Only replace words that exist in our dictionary
    - Keep sentence structure completely intact
    """
    # Strip the language tag (<2hi>) before processing
    tag_match = re.match(r'^(<2\w+>)\s+', text)
    tag       = tag_match.group(1) if tag_match else ""
    content   = text[tag_match.end():] if tag_match else text

    words = content.split()

    # Skip sentences that are too short
    if len(words) < MIN_SENTENCE_LENGTH:
        return text

    # Find candidate words (long enough, not Latin, in dictionary)
    candidates = [
        i for i, w in enumerate(words)
        if len(w) >= 3
        and not re.match(r'^[a-zA-Z]+$', w)   # not already English/Latin
        and w in dictionary
    ]

    if not candidates:
        return text

    # Pick how many to replace (20-40% of candidates)
    replace_prob = random.uniform(MIN_REPLACE_RATIO, MAX_REPLACE_RATIO)

    for i in candidates:
        if random.random() < replace_prob:
            words[i] = dictionary[words[i]]

    mixed = " ".join(words)
    return f"{tag} {mixed}".strip() if tag else mixed


def has_mixed_script(text):
    """Returns True if text has BOTH Latin and Indic script characters."""
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    has_indic = bool(re.search(r'[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF]', text))
    return has_latin and has_indic


class SyntheticDataGenerator:
    def __init__(self):
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Load all dictionaries from artifacts/data/dictionaries/
            self.dictionaries = {
                lang: load_dictionary(lang)
                for lang in SUPPORTED_LANGUAGES
            }

            logging.info("SyntheticDataGenerator ready")
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def generate_for_split(self, input_path, output_path, language):
        """
        Reads a preprocessed CSV, applies code-mixing to text column,
        keeps only rows where mixing actually happened, saves result.
        """
        try:
            df = pd.read_csv(input_path)
            logging.info(f"Generating synthetic data [{language}]: {len(df)} rows")

            dictionary = self.dictionaries[language]

            # Apply mixing to text column only (summary stays clean)
            df["text"] = df["text"].apply(lambda t: mix_sentence(t, dictionary))

            # Keep only rows where mixing actually produced Indic+Latin mix
            before = len(df)
            df = df[df["text"].apply(has_mixed_script)].reset_index(drop=True)
            logging.info(f"  Mixed: {len(df)}/{before} rows had successful mixing")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logging.info(f"  Saved → {output_path}")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def run(self):
        """
        Generates synthetic code-mixed data for Hindi, Bengali, Gujarati.
        English is skipped (no mixing needed for English text).
        Reads from preprocessed/, saves to synthetic_codemixed/.
        """
        try:
            logging.info("Starting synthetic data generation...")
            random.seed(42)

            for lang in SUPPORTED_LANGUAGES:
                logging.info(f"\nProcessing [{lang}]...")
                for split in ["train", "val", "test"]:
                    input_path  = os.path.join(DATA_DIR, "preprocessed", lang, f"{split}_clean.csv")
                    output_path = os.path.join(OUTPUT_DIR, lang, f"{split}_synthetic.csv")

                    if not os.path.exists(input_path):
                        logging.warning(f"Not found, skipping: {input_path}")
                        continue

                    self.generate_for_split(input_path, output_path, lang)

            logging.info("Synthetic data generation complete!")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.run()
    print("Synthetic data generation complete!")
    print("Output saved to artifacts/data/synthetic_codemixed/")