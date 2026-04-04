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

    # ------------------------------------------------------------------ #
    #  ILSUM-2.0 — monolingual summarization (Phase 1 training base)      #
    # ------------------------------------------------------------------ #
    ILSUM_DATASET_ID = "ILSUM/ILSUM-2.0"
    ILSUM_LANGUAGES  = ["Hindi", "Bengali", "English", "Gujarati"]

    # Raw ILSUM-2.0 column names → our standard names
    ILSUM_COLUMN_MAP = {
        "Article": "text",
        "Summary": "summary",
        "Heading": "heading",   # dropped after rename
    }

    # ------------------------------------------------------------------ #
    #  HinGE — Hindi↔Hinglish translation dataset (Phase 2 code-mixed)   #
    #  ID    : LingoIITGN/HinGE                                           #
    #  Split : train only (1.98k rows)                                    #
    #  Cols  : English, Hindi, Human-generated Hinglish,                  #
    #          WAC, WAC rating1, WAC rating2,                             #
    #          PAC, PAC rating1, PAC rating2                              #
    #                                                                     #
    #  We use:                                                            #
    #    Hindi                    → text    (monolingual source)          #
    #    Human-generated Hinglish → summary (code-mixed target)           #
    # ------------------------------------------------------------------ #
    HINGE_DATASET_ID  = "LingoIITGN/HinGE"
    HINGE_COLUMN_MAP  = {
        "Hindi":                    "text",
        "Human-generated Hinglish": "summary",
    }
    # Columns to keep after rename (drop ratings, WAC, PAC, English)
    HINGE_KEEP_COLS   = ["text", "summary"]

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
    #  ILSUM loader                                                        #
    # ------------------------------------------------------------------ #
    def load_ilsum_dataset(self, languages: list = None) -> pd.DataFrame:
        """
        Load ILSUM-2.0 from HuggingFace across all (or selected) language configs.

        Raw columns:
            id       → dropped
            Article  → text
            Summary  → summary
            Heading  → dropped

        Each language config has 'train' and 'test' splits.
        All splits are pooled; we re-split into train/val/test ourselves.

        Args:
            languages: Language configs to load. Defaults to all four.

        Returns:
            DataFrame with columns: 'text', 'summary', 'language'
        """
        try:
            languages = languages or self.ILSUM_LANGUAGES
            logging.info(f"Loading ILSUM-2.0 — languages: {languages}")

            all_data = []
            for lang in languages:
                try:
                    dataset = load_dataset(self.ILSUM_DATASET_ID, lang)
                    logging.info(f"  [{lang}] splits: {list(dataset.keys())}")

                    for split_name, split_data in dataset.items():
                        df = split_data.to_pandas()
                        logging.info(f"  [{lang}/{split_name}] "
                                     f"cols: {list(df.columns)}, rows: {len(df)}")

                        df = df.rename(columns=self.ILSUM_COLUMN_MAP)

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
                        all_data.append(df)

                    total = sum(len(dataset[s]) for s in dataset)
                    logging.info(f"  [{lang}] {total} total rows loaded")

                except Exception as lang_err:
                    logging.error(f"  [{lang}] FAILED: {lang_err}")
                    raise CodeMixedSummarizationException(lang_err, sys)

            ilsum_df = pd.concat(all_data, ignore_index=True)

            before   = len(ilsum_df)
            ilsum_df = ilsum_df.dropna(subset=["text", "summary"]).reset_index(drop=True)
            dropped  = before - len(ilsum_df)
            if dropped:
                logging.warning(f"Dropped {dropped} null rows from ILSUM")

            logging.info(f"ILSUM-2.0 ready — {len(ilsum_df)} rows, "
                         f"cols: {list(ilsum_df.columns)}")
            return ilsum_df

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  HinGE loader                                                        #
    # ------------------------------------------------------------------ #
    def load_hinge_dataset(self) -> pd.DataFrame:
        """
        Load HinGE (LingoIITGN/HinGE) from HuggingFace.

        Raw columns used:
            Hindi                    → text    (monolingual Hindi source)
            Human-generated Hinglish → summary (code-mixed Hinglish target)

        All other columns (English, WAC, WAC rating1/2, PAC, PAC rating1/2)
        are dropped — they are quality ratings not needed for training.

        Returns:
            DataFrame with columns: 'text', 'summary', 'language'
        """
        try:
            logging.info(f"Loading HinGE — id: {self.HINGE_DATASET_ID}")

            dataset   = load_dataset(self.HINGE_DATASET_ID)
            split_key = "train" if "train" in dataset else list(dataset.keys())[0]
            df        = dataset[split_key].to_pandas()

            logging.info(f"  [HinGE/{split_key}] raw cols: {list(df.columns)}, "
                         f"rows: {len(df)}")

            # Validate raw columns exist before rename
            for raw_col in self.HINGE_COLUMN_MAP:
                if raw_col not in df.columns:
                    raise ValueError(
                        f"[HinGE] Expected column '{raw_col}' not found. "
                        f"Available: {list(df.columns)}"
                    )

            df = df.rename(columns=self.HINGE_COLUMN_MAP)

            # Keep only text + summary, drop all rating/other columns
            df = df[self.HINGE_KEEP_COLS].copy()

            df["language"] = "Hinglish"

            before = len(df)
            df     = df.dropna(subset=["text", "summary"]).reset_index(drop=True)
            dropped = before - len(df)
            if dropped:
                logging.warning(f"Dropped {dropped} null rows from HinGE")

            logging.info(f"HinGE ready — {len(df)} rows, cols: {list(df.columns)}")
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
        """Split a DataFrame into train / val / test sets."""
        try:
            assert round(train_ratio + val_ratio + test_ratio, 5) == 1.0, \
                "Ratios must sum to 1.0"

            logging.info(f"Splitting {len(dataframe)} rows — "
                         f"train:{train_ratio} val:{val_ratio} test:{test_ratio}")

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

            logging.info(f"Split — Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")
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
        """Save train/val/test DataFrames as CSVs to save_dir."""
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
        """
        Main pipeline: load all datasets, split, and save to artifacts/.

        Datasets loaded:
            ilsum  → artifacts/data/ilsum_processed/   (Phase 1 base)
            hinge  → artifacts/data/real_codemixed/    (Phase 2 code-mixed)

        Returns:
            Dict mapping dataset name → { 'train': path, 'val': path, 'test': path }
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting data loading pipeline...")
            logging.info("=" * 50)

            artifact = {}

            # --- ILSUM (monolingual — Phase 1) ---
            logging.info("Step 1/2: Loading ILSUM dataset...")
            ilsum_df = self.load_ilsum_dataset()
            artifact["ilsum"] = self.save_splits(
                *self.split_data(ilsum_df),
                save_dir=self.ilsum_dir
            )

            # --- HinGE (code-mixed — Phase 2) ---
            logging.info("Step 2/2: Loading HinGE dataset...")
            hinge_df = self.load_hinge_dataset()
            artifact["hinge"] = self.save_splits(
                *self.split_data(hinge_df),
                save_dir=self.codemixed_dir
            )

            logging.info("=" * 50)
            logging.info("Data loading pipeline completed successfully!")
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
        for dataset_name, paths in artifacts.items():
            print(f"\n{dataset_name.upper()}:")
            for split, path in paths.items():
                print(f"  {split}: {path}")

        print("\nData loading completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()