
"""
Cleans input text: removes extra spaces, strips noisy characters, normalizes
unicode, and prepends the appropriate language tag (<2hi>, <2ta>, <2en>).
"""

import sys
import os
import re
import unicodedata
import pandas as pd
from typing import Dict

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


class TextPreprocessor:
    """
    Cleans and prepares text data for IndicBART training and inference.

    Responsibilities:
        - Unicode normalization (NFC) for Indic scripts
        - Remove URLs, HTML entities, control characters
        - Remove zero-width characters common in Indic text
        - Normalize whitespace
        - Prepend IndicBART language tags to text column ONLY

    Language tag mapping (IndicBART official):
        Hindi     → <2hi>
        Bengali   → <2bn>
        English   → <2en>
        Gujarati  → <2gu>
        Hinglish  → <2hi>  (Hindi is dominant language in Hinglish)

    Tag is prepended ONLY to 'text' column (model input).
    'summary' column is cleaned but never tagged (it is the target output).

    Modes:
        'train'     → cleans both text and summary columns
        'inference' → cleans only text column (no summary exists)
    """

    # ------------------------------------------------------------------ #
    #  IndicBART official language tag mapping                            #
    #  Source: ai4bharat/IndicBART model card                            #
    #  Format: "<2xx> text..." (tag + single space + text)               #
    # ------------------------------------------------------------------ #
    LANGUAGE_TAG_MAP: Dict[str, str] = {
        "Hindi":    "<2hi>",
        "Bengali":  "<2bn>",
        "English":  "<2en>",
        "Gujarati": "<2gu>",
        "Hinglish": "<2hi>",   # Hindi dominant in code-mixed Hinglish
    }

    # ------------------------------------------------------------------ #
    #  Regex patterns compiled once at class level for efficiency         #
    # ------------------------------------------------------------------ #

    # URLs: http/https/www patterns
    _URL_PATTERN = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|"
        r"(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[^\s]+"
    )

    # HTML entities: &amp; &lt; &gt; &nbsp; &#123; etc.
    _HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#[0-9]+;")

    # HTML/XML tags: <b>, </div>, <br/> etc.
    _HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

    # Control characters (except newline \n and tab \t)
    _CONTROL_CHAR_PATTERN = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
    )

    # Zero-width characters common in Indic scripts:
    #   \u200b Zero Width Space
    #   \u200c Zero Width Non-Joiner (ZWNJ)
    #   \u200d Zero Width Joiner (ZWJ)
    #   \ufeff Byte Order Mark (BOM)
    #   \u00ad Soft Hyphen
    _ZERO_WIDTH_PATTERN = re.compile(
        r"[\u200b\u200c\u200d\ufeff\u00ad]"
    )

    # Multiple whitespace (spaces, tabs, newlines) → single space
    _WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        output_dir: str = "artifacts/data/preprocessed",
    ):
        """
        Initialize TextPreprocessor.

        Args:
            output_dir: Root directory to save preprocessed CSVs.
                        Subfolders mirror the ilsum_processed/ structure:
                        artifacts/data/preprocessed/
                        ├── Hindi/train_clean.csv ...
                        ├── Bengali/
                        ├── English/
                        ├── Gujarati/
                        └── real_codemixed/train_clean.csv ...
        """
        try:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"TextPreprocessor initialized — output_dir: {output_dir}")
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Private cleaning helpers (operate on a single string)              #
    # ------------------------------------------------------------------ #

    def _normalize_unicode(self, text: str) -> str:
        """
        Apply NFC unicode normalization.
        Indic scripts often have multiple byte representations for the
        same visible character. NFC collapses these into one canonical form,
        preventing tokenizer mismatches.
        """
        return unicodedata.normalize("NFC", text)

    def _remove_urls(self, text: str) -> str:
        """Remove http/https/www URLs — common in ILSUM news articles."""
        return self._URL_PATTERN.sub(" ", text)

    def _remove_html(self, text: str) -> str:
        """Remove HTML entities (&amp;) and HTML/XML tags (<b>)."""
        text = self._HTML_ENTITY_PATTERN.sub(" ", text)
        text = self._HTML_TAG_PATTERN.sub(" ", text)
        return text

    def _remove_control_characters(self, text: str) -> str:
        """
        Remove invisible control characters.
        Keeps newline and tab which may carry meaning in article structure,
        but collapses them into spaces in the whitespace step.
        """
        return self._CONTROL_CHAR_PATTERN.sub(" ", text)

    def _remove_zero_width_chars(self, text: str) -> str:
        """
        Remove zero-width characters specific to Indic scripts.
        ZWJ/ZWNJ affect rendering but cause tokenizer inconsistencies
        when not properly handled by the model's vocabulary.
        """
        return self._ZERO_WIDTH_PATTERN.sub("", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse all whitespace sequences into a single space and strip."""
        return self._WHITESPACE_PATTERN.sub(" ", text).strip()

    def _clean_text(self, text: str) -> str:
        """
        Apply full cleaning pipeline to a single string.
        Order matters — HTML removal before whitespace normalization,
        unicode normalization first to ensure regex patterns work correctly.

        Pipeline:
            1. Unicode NFC normalization
            2. Remove zero-width characters
            3. Remove URLs
            4. Remove HTML tags and entities
            5. Remove control characters
            6. Normalize whitespace
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._normalize_unicode(text)
        text = self._remove_zero_width_chars(text)
        text = self._remove_urls(text)
        text = self._remove_html(text)
        text = self._remove_control_characters(text)
        text = self._normalize_whitespace(text)
        return text

    def _get_language_tag(self, language: str) -> str:
        """
        Look up the IndicBART language tag for a given language name.

        Args:
            language: Language name as stored in the 'language' column
                      e.g. 'Hindi', 'Bengali', 'Hinglish'

        Returns:
            IndicBART tag string e.g. '<2hi>'

        Raises:
            ValueError if language is not in LANGUAGE_TAG_MAP
        """
        if language not in self.LANGUAGE_TAG_MAP:
            raise ValueError(
                f"Unknown language '{language}'. "
                f"Supported: {list(self.LANGUAGE_TAG_MAP.keys())}"
            )
        return self.LANGUAGE_TAG_MAP[language]

    def _prepend_language_tag(self, text: str, language: str) -> str:
        """
        Prepend the IndicBART language tag to text.
        Format: "<2hi> text..." (tag + single space + text)

        ONLY applied to the 'text' column (model input).
        NEVER applied to the 'summary' column (model target output).

        Args:
            text    : Cleaned input text
            language: Language name from the 'language' column

        Returns:
            Tagged text string e.g. "<2hi> आज मौसम अच्छा है"
        """
        tag = self._get_language_tag(language)
        return f"{tag} {text}"

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        df: pd.DataFrame,
        mode: str = "train",
    ) -> pd.DataFrame:
        """
        Clean a DataFrame and prepend language tags to the text column.

        Example:
            BEFORE:
                text    : "आज बहुत अच्छा&nbsp;मौसम  है  http://news.com"
                summary : "आज मौसम अच्छा  है"
                language: "Hindi"

            AFTER (train mode):
                text    : "<2hi> आज बहुत अच्छा मौसम है"
                summary : "आज मौसम अच्छा है"
                language: "Hindi"
        """
        try:
            if mode not in ("train", "inference"):
                raise ValueError(
                    f"Invalid mode '{mode}'. Choose 'train' or 'inference'."
                )

            # Validate required columns
            required = ["text", "language"]
            if mode == "train":
                required.append("summary")
            for col in required:
                if col not in df.columns:
                    raise ValueError(
                        f"Column '{col}' missing from DataFrame. "
                        f"Available: {list(df.columns)}"
                    )

            logging.info(
                f"Preprocessing {len(df)} rows in '{mode}' mode — "
                f"languages: {df['language'].unique().tolist()}"
            )

            df = df.copy()

            # Step 1: Clean text column
            logging.info("  Cleaning 'text' column...")
            df["text"] = df["text"].apply(self._clean_text)

            # Step 2: Clean summary column (training only)
            if mode == "train":
                logging.info("  Cleaning 'summary' column...")
                df["summary"] = df["summary"].apply(self._clean_text)

            # Step 3: Prepend language tag to text ONLY
            logging.info("  Prepending language tags to 'text' column...")
            df["text"] = df.apply(
                lambda row: self._prepend_language_tag(row["text"], row["language"]),
                axis=1
            )

            # Step 4: Drop rows where text or summary became empty after cleaning
            before = len(df)
            if mode == "train":
                df = df[
                    df["text"].str.strip().astype(bool) &
                    df["summary"].str.strip().astype(bool)
                ].reset_index(drop=True)
            else:
                df = df[
                    df["text"].str.strip().astype(bool)
                ].reset_index(drop=True)

            dropped = before - len(df)
            if dropped:
                logging.warning(f"  Dropped {dropped} rows that became empty after cleaning")

            logging.info(f"  Preprocessing complete — {len(df)} rows remaining")
            return df

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def preprocess_and_save(
        self,
        df: pd.DataFrame,
        save_path: str,
        mode: str = "train",
    ) -> str:
        """
        Preprocess a DataFrame and save the result as a CSV.

        Args:
            df        : Input DataFrame
            save_path : Full path to save the cleaned CSV
                        e.g. 'artifacts/data/preprocessed/Hindi/train_clean.csv'
            mode      : 'train' or 'inference'

        Returns:
            Path where the cleaned CSV was saved
        """
        try:
            cleaned_df = self.preprocess(df, mode=mode)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cleaned_df.to_csv(save_path, index=False)

            logging.info(f"  Saved preprocessed data: {save_path} ({len(cleaned_df)} rows)")
            return save_path

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def preprocess_ilsum_splits(
        self,
        split_paths: Dict[str, str],
        language: str,
    ) -> Dict[str, str]:
        """
        Preprocess all three splits (train/val/test) for one ILSUM language.
        Reads CSVs from split_paths, cleans them, saves to preprocessed/.

        Args:
            split_paths : Dict from dataset_loader output e.g.:
                          { 'train': 'artifacts/data/ilsum_processed/Hindi/train.csv',
                            'val'  : '...val.csv',
                            'test' : '...test.csv' }
            language    : Language name e.g. 'Hindi'

        Returns:
            Dict mapping split → cleaned CSV path e.g.:
            { 'train': 'artifacts/data/preprocessed/Hindi/train_clean.csv', ... }
        """
        try:
            logging.info(f"Preprocessing ILSUM [{language}] splits...")
            output_paths = {}

            for split_name, input_path in split_paths.items():
                logging.info(f"  [{language}/{split_name}] reading: {input_path}")
                df = pd.read_csv(input_path)

                save_path = os.path.join(
                    self.output_dir,
                    language,
                    f"{split_name}_clean.csv"
                )
                output_paths[split_name] = self.preprocess_and_save(
                    df, save_path, mode="train"
                )

            logging.info(f"  [{language}] all splits preprocessed")
            return output_paths

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def preprocess_codemixed_splits(
        self,
        split_paths: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Preprocess all three splits for the HinGE code-mixed dataset.
        Reads from real_codemixed/, saves to preprocessed/real_codemixed/.

        Args:
            split_paths: Dict from dataset_loader output e.g.:
                         { 'train': 'artifacts/data/real_codemixed/train.csv', ... }

        Returns:
            Dict mapping split → cleaned CSV path
        """
        try:
            logging.info("Preprocessing HinGE code-mixed splits...")
            output_paths = {}

            for split_name, input_path in split_paths.items():
                logging.info(f"  [HinGE/{split_name}] reading: {input_path}")
                df = pd.read_csv(input_path)

                save_path = os.path.join(
                    self.output_dir,
                    "real_codemixed",
                    f"{split_name}_clean.csv"
                )
                output_paths[split_name] = self.preprocess_and_save(
                    df, save_path, mode="train"
                )

            logging.info("  HinGE all splits preprocessed")
            return output_paths

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def initiate_preprocessing(
        self,
        data_artifacts: Dict,
    ) -> Dict:
        """
        Main pipeline entry point. Preprocesses all datasets produced
        by dataset_loader.initiate_data_loading().

        """
        try:
            logging.info("=" * 50)
            logging.info("Starting preprocessing pipeline...")
            logging.info("=" * 50)

            preprocessed_artifacts = {"ilsum": {}, "hinge": {}}

            # --- Step 1: Preprocess each ILSUM language ---
            logging.info("Step 1/2: Preprocessing ILSUM (per language)...")
            for lang, split_paths in data_artifacts["ilsum"].items():
                preprocessed_artifacts["ilsum"][lang] = \
                    self.preprocess_ilsum_splits(split_paths, lang)

            # --- Step 2: Preprocess HinGE code-mixed ---
            logging.info("Step 2/2: Preprocessing HinGE code-mixed data...")
            preprocessed_artifacts["hinge"] = \
                self.preprocess_codemixed_splits(data_artifacts["hinge"])

            logging.info("=" * 50)
            logging.info("Preprocessing pipeline completed successfully!")
            logging.info("Preprocessed artifact summary:")
            for lang, paths in preprocessed_artifacts["ilsum"].items():
                logging.info(f"  ILSUM/{lang}: {paths}")
            logging.info(f"  HinGE: {preprocessed_artifacts['hinge']}")
            logging.info("=" * 50)

            return preprocessed_artifacts

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    try:
        from src.components.data.dataset_loader import DatasetLoader

        print("Step 1: Loading datasets...")
        loader        = DatasetLoader()
        data_artifacts = loader.initiate_data_loading()

        print("\nStep 2: Preprocessing...")
        preprocessor  = TextPreprocessor()
        clean_artifacts = preprocessor.initiate_preprocessing(data_artifacts)

        print("\n" + "=" * 50)
        print("Preprocessed Artifacts:")
        print("=" * 50)

        print("\nILSUM (per language):")
        for lang, paths in clean_artifacts["ilsum"].items():
            print(f"  {lang}:")
            for split, path in paths.items():
                print(f"    {split}: {path}")

        print("\nHinGE (code-mixed):")
        for split, path in clean_artifacts["hinge"].items():
            print(f"  {split}: {path}")

        print("\nPreprocessing completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()