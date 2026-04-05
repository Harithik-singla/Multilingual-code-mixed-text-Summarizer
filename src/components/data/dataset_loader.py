"""
Loads ILSUM dataset and real code-mixed datasets (HinGE) using
HuggingFace `datasets`. Returns train/val/test splits.
"""

import sys
import os
import pandas as pd
from typing import Tuple, Dict
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


class DatasetLoader:
    """Load and preprocess ILSUM and HinGE code-mixed datasets."""

    ILSUM_DATASET_ID = "ILSUM/ILSUM-2.0"
    ILSUM_LANGUAGES  = ["Hindi", "Bengali", "English", "Gujarati"]
    ILSUM_COLUMN_MAP = {
        "Article": "text",
        "Summary": "summary",
        "Heading": "heading",  # renamed then dropped
    }

   
    HINGE_DATASET_ID = "LingoIITGN/HinGE"
    HINGE_COLUMN_MAP = {
        "Human-generated Hinglish": "text",    # code-mixed input
        "English":                  "summary", # clean English output
    }
    HINGE_KEEP_COLS = ["text", "summary", "language"]

    def __init__(self, data_dir: str = "artifacts/data"):
        try:
            self.data_dir      = data_dir
            self.ilsum_dir     = os.path.join(data_dir, "ilsum_processed")
            self.codemixed_dir = os.path.join(data_dir, "real_codemixed")

            os.makedirs(self.ilsum_dir, exist_ok=True)
            os.makedirs(self.codemixed_dir, exist_ok=True)

            logging.info(f"DatasetLoader initialized — data_dir: {data_dir}")
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  ILSUM — load a single language                                      #
    # ------------------------------------------------------------------ #
    def load_ilsum_language(self, lang: str) -> pd.DataFrame:
        """
        Load one ILSUM-2.0 language config from HuggingFace.
        Pools its train + test splits into one DataFrame.
        We re-split into train/val/test ourselves via split_data().

        Args:
            lang: One of 'Hindi', 'Bengali', 'English', 'Gujarati'

        Returns:
            DataFrame with columns: 'text', 'summary', 'language'
        """
        try:
            if lang not in self.ILSUM_LANGUAGES:
                raise ValueError(
                    f"Unknown language '{lang}'. "
                    f"Choose from: {self.ILSUM_LANGUAGES}"
                )

            logging.info(f"Loading ILSUM-2.0 [{lang}]...")
            dataset = load_dataset(self.ILSUM_DATASET_ID, lang)
            logging.info(f"  [{lang}] splits found: {list(dataset.keys())}")

            all_splits = []
            for split_name, split_data in dataset.items():
                df = split_data.to_pandas()
                logging.info(
                    f"  [{lang}/{split_name}] "
                    f"raw cols: {list(df.columns)}, rows: {len(df)}"
                )

                df = df.rename(columns=self.ILSUM_COLUMN_MAP)

                # Validate required columns exist after rename
                for required in ["text", "summary"]:
                    if required not in df.columns:
                        raise ValueError(
                            f"[{lang}/{split_name}] '{required}' missing after rename. "
                            f"Available: {list(df.columns)}"
                        )

                df["language"] = lang
                df = df.drop(
                    columns=[c for c in ["id", "heading"] if c in df.columns]
                )
                all_splits.append(df)

            lang_df = pd.concat(all_splits, ignore_index=True)

            # Drop null rows
            before  = len(lang_df)
            lang_df = lang_df.dropna(subset=["text", "summary"]).reset_index(drop=True)
            dropped = before - len(lang_df)
            if dropped:
                logging.warning(f"  [{lang}] Dropped {dropped} null rows")

            logging.info(f"  [{lang}] ready — {len(lang_df)} total rows")
            return lang_df

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def load_all_ilsum_languages(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 4 ILSUM-2.0 language configs individually.
        Each language stays as a separate DataFrame for
        per-language Phase 1 training.

        Returns:
            Dict mapping language name → DataFrame
            e.g. { 'Hindi': df_hi, 'Bengali': df_bn, ... }
        """
        try:
            logging.info(f"Loading all ILSUM languages: {self.ILSUM_LANGUAGES}")
            language_dfs = {}

            for lang in self.ILSUM_LANGUAGES:
                language_dfs[lang] = self.load_ilsum_language(lang)

            logging.info("ILSUM load summary:")
            for lang, df in language_dfs.items():
                logging.info(f"  {lang}: {len(df)} rows")

            return language_dfs

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  HinGE loader                                                        #
    # ------------------------------------------------------------------ #
    def load_hinge_dataset(self) -> pd.DataFrame:
        """
        Load HinGE (LingoIITGN/HinGE) from HuggingFace.

        Mapping rationale:
            Human-generated Hinglish → text
                Code-mixed sentences the model must learn to READ.
                This is the noisy real-world input users write.

            English → summary
                Clean English equivalent the model must PRODUCE.
                Matches Phase 1 output style (clean, monolingual).

        This direction (Hinglish → English) teaches the model to:
            1. Understand code-mixed Hinglish input
            2. Still produce clean English output
        ...which is exactly the Phase 2 code-mixed adaptation goal.

        Returns:
            DataFrame with columns: 'text', 'summary', 'language'
        """
        try:
            logging.info(f"Loading HinGE — id: {self.HINGE_DATASET_ID}")

            dataset   = load_dataset(self.HINGE_DATASET_ID)
            split_key = "train" if "train" in dataset else list(dataset.keys())[0]
            df        = dataset[split_key].to_pandas()

            logging.info(
                f"  [HinGE/{split_key}] "
                f"raw cols: {list(df.columns)}, rows: {len(df)}"
            )

            # Validate expected raw columns exist before rename
            for raw_col in self.HINGE_COLUMN_MAP:
                if raw_col not in df.columns:
                    raise ValueError(
                        f"[HinGE] Expected column '{raw_col}' not found. "
                        f"Available: {list(df.columns)}"
                    )

            df = df.rename(columns=self.HINGE_COLUMN_MAP)

            # Add language tag then keep only needed columns
            df["language"] = "Hinglish"
            df = df[self.HINGE_KEEP_COLS].copy()

            # Drop null rows
            before  = len(df)
            df      = df.dropna(subset=["text", "summary"]).reset_index(drop=True)
            dropped = before - len(df)
            if dropped:
                logging.warning(f"  [HinGE] Dropped {dropped} null rows")

            # Log a sample so we can visually confirm mapping is correct
            logging.info(
                f"HinGE ready — {len(df)} rows\n"
                f"  Sample text    (Hinglish): {df['text'].iloc[0][:80]}...\n"
                f"  Sample summary (English) : {df['summary'].iloc[0][:80]}..."
            )
            return df

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Split                                                               #
    # ------------------------------------------------------------------ #
    def split_data(
        self,
        dataframe: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float   = 0.15,
        test_ratio: float  = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a DataFrame into train / val / test sets.

        Args:
            dataframe   : Input DataFrame to split
            train_ratio : Fraction for training   (default 0.70)
            val_ratio   : Fraction for validation (default 0.15)
            test_ratio  : Fraction for testing    (default 0.15)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        try:
            assert round(train_ratio + val_ratio + test_ratio, 5) == 1.0, \
                "train_ratio + val_ratio + test_ratio must equal 1.0"

            logging.info(
                f"Splitting {len(dataframe)} rows — "
                f"train:{train_ratio} val:{val_ratio} test:{test_ratio}"
            )

            train_df, temp_df = train_test_split(
                dataframe,
                test_size=round(val_ratio + test_ratio, 5),
                random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=round(test_ratio / (val_ratio + test_ratio), 5),
                random_state=42
            )

            logging.info(
                f"Split done — "
                f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
            )
            return train_df, val_df, test_df

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        save_dir: str,
    ) -> Dict[str, str]:
        """
        Save train/val/test DataFrames as CSVs to save_dir.

        Args:
            train_df : Training DataFrame
            val_df   : Validation DataFrame
            test_df  : Test DataFrame
            save_dir : Directory to save CSVs into

        Returns:
            Dict mapping split name → csv file path
            e.g. { 'train': '.../train.csv', 'val': '...', 'test': '...' }
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            paths = {}
            for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                path = os.path.join(save_dir, f"{split_name}.csv")
                df.to_csv(path, index=False)
                paths[split_name] = path
                logging.info(f"  Saved {split_name}: {path} ({len(df)} rows)")
            return paths

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Main pipeline                                                       #
    # ------------------------------------------------------------------ #
    def initiate_data_loading(self) -> Dict[str, Dict[str, str]]:
       
        try:
            logging.info("=" * 50)
            logging.info("Starting data loading pipeline...")
            logging.info("=" * 50)

            artifact = {"ilsum": {}, "hinge": {}}

            # --- Step 1: ILSUM — one subfolder per language ---
            logging.info("Step 1/2: Loading ILSUM (per language)...")
            language_dfs = self.load_all_ilsum_languages()

            for lang, lang_df in language_dfs.items():
                lang_save_dir = os.path.join(self.ilsum_dir, lang)
                logging.info(f"  Splitting and saving ILSUM [{lang}]...")
                artifact["ilsum"][lang] = self.save_splits(
                    *self.split_data(lang_df),
                    save_dir=lang_save_dir
                )

            # --- Step 2: HinGE — single folder for code-mixed ---
            logging.info("Step 2/2: Loading HinGE (Hinglish → English)...")
            hinge_df = self.load_hinge_dataset()
            artifact["hinge"] = self.save_splits(
                *self.split_data(hinge_df),
                save_dir=self.codemixed_dir
            )

            logging.info("=" * 50)
            logging.info("Data loading pipeline completed successfully!")
            logging.info("Artifact summary:")
            for lang, paths in artifact["ilsum"].items():
                logging.info(f"  ILSUM/{lang}: {paths}")
            logging.info(f"  HinGE: {artifact['hinge']}")
            logging.info("=" * 50)

            return artifact

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    try:
        print("Starting dataset loading pipeline...")
        loader    = DatasetLoader()
        artifacts = loader.initiate_data_loading()

        print("\n" + "=" * 50)
        print("Data Loading Artifacts:")
        print("=" * 50)

        print("\nILSUM (per language):")
        for lang, paths in artifacts["ilsum"].items():
            print(f"  {lang}:")
            for split, path in paths.items():
                print(f"    {split}: {path}")

        print("\nHinGE (Hinglish → English):")
        for split, path in artifacts["hinge"].items():
            print(f"  {split}: {path}")

        print("\nData loading completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()