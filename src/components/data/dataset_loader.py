"""
Downloads ILSUM-2.0 and HinGE datasets from HuggingFace,
splits into train/val/test (70/15/15), and saves as CSV files.

Run this once to get all data on disk:
    python -m src.components.data.dataset_loader
"""

import sys
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


ILSUM_LANGUAGES = ["Hindi", "Bengali", "English", "Gujarati"]
DATA_DIR        = "artifacts/data"


def split_and_save(df, save_dir):
    """Splits a DataFrame into train/val/test and saves as CSVs."""
    os.makedirs(save_dir, exist_ok=True)

    # 70% train, 30% temp
    train, temp = train_test_split(df, test_size=0.30, random_state=42)
    # Split temp into 50/50 → 15% val, 15% test
    val, test   = train_test_split(temp, test_size=0.50, random_state=42)

    train.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val.to_csv(  os.path.join(save_dir, "val.csv"),   index=False)
    test.to_csv( os.path.join(save_dir, "test.csv"),  index=False)

    logging.info(f"Saved to {save_dir} — train:{len(train)}, val:{len(val)}, test:{len(test)}")


class DatasetLoader:
    def __init__(self):
        try:
            self.ilsum_dir     = os.path.join(DATA_DIR, "ilsum_processed")
            self.codemixed_dir = os.path.join(DATA_DIR, "real_codemixed")
            os.makedirs(self.ilsum_dir,     exist_ok=True)
            os.makedirs(self.codemixed_dir, exist_ok=True)
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def load_ilsum_language(self, lang):
        """Downloads one ILSUM language, renames columns, returns a DataFrame."""
        try:
            logging.info(f"Loading ILSUM [{lang}]...")
            dataset = load_dataset("ILSUM/ILSUM-2.0", lang)

            # Combine all splits from HuggingFace (train+test)
            # We'll re-split ourselves
            dfs = []
            for split_data in dataset.values():
                df = split_data.to_pandas()
                df = df.rename(columns={"Article": "text", "Summary": "summary"})
                df["language"] = lang
                # Drop columns we don't need
                df = df[["text", "summary", "language"]]
                dfs.append(df)

            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.dropna(subset=["text", "summary"]).reset_index(drop=True)

            logging.info(f"  [{lang}] {len(combined)} rows loaded")
            return combined

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def load_hinge(self):
        """Downloads HinGE (Hinglish↔English) dataset, returns a DataFrame."""
        try:
            logging.info("Loading HinGE dataset...")
            dataset = load_dataset("LingoIITGN/HinGE")

            # Use whichever split exists
            split_key = list(dataset.keys())[0]
            df = dataset[split_key].to_pandas()

            # Human-generated Hinglish = our input text
            # English = the clean output summary
            df = df.rename(columns={
                "Human-generated Hinglish": "text",
                "English": "summary"
            })
            df["language"] = "Hinglish"
            df = df[["text", "summary", "language"]]
            df = df.dropna(subset=["text", "summary"]).reset_index(drop=True)

            logging.info(f"HinGE loaded — {len(df)} rows")
            return df

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def run(self):
        """
        Downloads all datasets and saves train/val/test CSVs to disk.
        Call this once. After it runs, all data is available locally.
        """
        try:
            logging.info("Starting data loading pipeline...")

            # Load and save each ILSUM language separately
            for lang in ILSUM_LANGUAGES:
                df       = self.load_ilsum_language(lang)
                save_dir = os.path.join(self.ilsum_dir, lang)
                split_and_save(df, save_dir)

            # Load and save HinGE
            hinge_df = self.load_hinge()
            split_and_save(hinge_df, self.codemixed_dir)

            logging.info("Data loading pipeline complete!")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    loader = DatasetLoader()
    loader.run()
    print("All datasets downloaded and saved!")