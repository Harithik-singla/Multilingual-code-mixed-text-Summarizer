"""
Phase 1 Training Pipeline.

Runnable script: loads preprocessed ILSUM datasets for all 4 languages
(Hindi, Bengali, English, Gujarati), combines them, tokenizes using
IndicBART tokenizer, initializes ModelTrainer, runs Phase 1 monolingual
training, and saves checkpoint to artifacts/checkpoints/phase1/.

Run from project root:
    python -m src.pipelines.train_phase1
    OR
    python src/pipelines/train_phase1.py
"""

import sys
import os
import time
from typing import Dict, Optional

import pandas as pd
from datasets import concatenate_datasets, Dataset

from src.components.model.model_loader import ModelLoader
from src.components.model.tokenizer_utils import TokenizerUtils, INDICBART_LANG_TAGS
from src.components.model.trainer import ModelTrainer
from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #

PREPROCESSED_DIR   = "artifacts/data/preprocessed"
CHECKPOINT_DIR     = "artifacts/checkpoints/phase1"
MODEL_CONFIG_PATH  = "src/configs/model_config.yaml"
PHASE1_CONFIG_PATH = "src/configs/phase1_train.yaml"

# All ILSUM languages used in Phase 1 (English included — it's monolingual)
PHASE1_LANGUAGES   = ["Hindi", "Bengali", "English", "Gujarati"]


class Phase1TrainingPipeline:
    """
    Orchestrates the full Phase 1 monolingual training pipeline.

    Step-by-step:
        1. Load preprocessed ILSUM CSVs for all 4 languages
        2. Combine all languages into a single train + val DataFrame
        3. Load IndicBART model + tokenizer via ModelLoader
        4. Initialize TokenizerUtils (adds language tag special tokens)
        5. Tokenize combined train + val DataFrames → HuggingFace Datasets
        6. Initialize ModelTrainer with Phase 1 config
        7. Run training → save checkpoint to artifacts/checkpoints/phase1/
        8. Return training summary

    Why combine all 4 languages?
        Phase 1's goal is to teach the model SUMMARIZATION in general, not
        just Hindi. Exposing it to all 4 ILSUM languages simultaneously:
            - Gives more training examples (larger effective dataset)
            - Keeps the multilingual representations in IndicBART active
            - Prevents the model from forgetting Bengali/Gujarati/English
              when we fine-tune on Hindi-heavy code-mixed data in Phase 2

    Dataset size note:
        Hindi alone is ~128MB train. All 4 languages together can be
        large. Use max_samples_per_language to cap per-language rows
        for faster iteration / memory-constrained setups.
        Set to None to use ALL data.
    """

    def __init__(
        self,
        preprocessed_dir: str   = PREPROCESSED_DIR,
        model_config_path: str  = MODEL_CONFIG_PATH,
        phase_config_path: str  = PHASE1_CONFIG_PATH,
        languages: list         = None,
        max_samples_per_language: Optional[int] = None,
    ):
        """
        Initialize Phase1TrainingPipeline.

        Args:
            preprocessed_dir          : Root of preprocessed data artifacts.
            model_config_path         : Path to model_config.yaml.
            phase_config_path         : Path to phase1_train.yaml.
            languages                 : List of language names to include.
                                        Defaults to all 4 ILSUM languages.
            max_samples_per_language  : Cap rows loaded per language.
                                        Useful for quick testing.
                                        None = use all data (default).
        """
        try:
            self.preprocessed_dir           = preprocessed_dir
            self.model_config_path          = model_config_path
            self.phase_config_path          = phase_config_path
            self.languages                  = languages or PHASE1_LANGUAGES
            self.max_samples_per_language   = max_samples_per_language

            logging.info(
                f"Phase1TrainingPipeline initialized\n"
                f"  Languages           : {self.languages}\n"
                f"  max_samples_per_lang: {self.max_samples_per_language or 'ALL'}\n"
                f"  Preprocessed dir    : {self.preprocessed_dir}\n"
                f"  Phase config        : {self.phase_config_path}"
            )

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Step 1: Load & combine ILSUM CSVs                                  #
    # ------------------------------------------------------------------ #

    def _load_ilsum_split(self, split: str) -> pd.DataFrame:
        """
        Load a single data split (train / val / test) for all languages
        and combine them into one DataFrame.

        File naming convention from preprocessor.py:
            artifacts/data/preprocessed/Hindi/train_clean.csv
            artifacts/data/preprocessed/Bengali/val_clean.csv
            etc.

        Args:
            split : One of 'train', 'val', 'test'

        Returns:
            Combined DataFrame with rows from all languages for this split.
            Columns: text, summary, language
        """
        try:
            filename   = f"{split}_clean.csv"
            lang_dfs   = []

            for lang in self.languages:
                csv_path = os.path.join(
                    self.preprocessed_dir, lang, filename
                )

                if not os.path.exists(csv_path):
                    logging.warning(
                        f"  [{lang}/{split}] CSV not found: {csv_path} — skipping"
                    )
                    continue

                df = pd.read_csv(csv_path)

                # Optional: cap per-language rows
                if self.max_samples_per_language is not None:
                    df = df.head(self.max_samples_per_language)
                    logging.info(
                        f"  [{lang}/{split}] Capped to {len(df)} rows "
                        f"(max_samples_per_language={self.max_samples_per_language})"
                    )
                else:
                    logging.info(f"  [{lang}/{split}] Loaded {len(df)} rows")

                lang_dfs.append(df)

            if not lang_dfs:
                raise RuntimeError(
                    f"No data loaded for split='{split}'. "
                    f"Check that preprocessed CSVs exist in: {self.preprocessed_dir}"
                )

            combined = pd.concat(lang_dfs, ignore_index=True).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)
            # Shuffle after concat so languages are interleaved in batches,
            # not sorted (Hindi first, then Bengali, etc.) — interleaving
            # gives the model balanced gradient updates across languages.

            logging.info(
                f"[{split}] Combined {len(combined)} rows from "
                f"{len(lang_dfs)} languages"
            )
            return combined

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Main pipeline                                                       #
    # ------------------------------------------------------------------ #

    def run(self, resume_from_checkpoint: Optional[str] = None) -> Dict:
        """
        Execute the full Phase 1 training pipeline.

        Args:
            resume_from_checkpoint : Optional checkpoint path to resume from.
                                     e.g. 'artifacts/checkpoints/phase1/checkpoint-500'
                                     Pass None to start fresh (default).

        Returns:
            Dict with training summary:
            {
                'phase'      : 1,
                'output_dir' : 'artifacts/checkpoints/phase1',
                'train_loss' : float,
                'epochs'     : int,
                'languages'  : list,
                'train_size' : int,
                'val_size'   : int,
                'duration_min': float,
            }
        """
        try:
            start_time = time.time()

            logging.info("=" * 60)
            logging.info("PHASE 1 TRAINING PIPELINE STARTED")
            logging.info("=" * 60)

            # ----------------------------------------------------------
            # Step 1: Load & combine ILSUM CSVs
            # ----------------------------------------------------------
            logging.info("\nStep 1/5: Loading ILSUM preprocessed data...")
            train_df = self._load_ilsum_split("train")
            val_df   = self._load_ilsum_split("val")

            logging.info(
                f"  Train size : {len(train_df)} rows\n"
                f"  Val size   : {len(val_df)} rows\n"
                f"  Languages  : {train_df['language'].value_counts().to_dict()}"
            )

            # ----------------------------------------------------------
            # Step 2: Load IndicBART model + tokenizer
            # ----------------------------------------------------------
            logging.info("\nStep 2/5: Loading IndicBART model + tokenizer...")
            loader             = ModelLoader(config_path=self.model_config_path)
            model, tokenizer, device = loader.load()
            model_config       = loader.get_config()

            logging.info(f"  Model loaded on device: {device}")

            # ----------------------------------------------------------
            # Step 3: Initialize TokenizerUtils (adds special tokens)
            # ----------------------------------------------------------
            logging.info("\nStep 3/5: Initializing TokenizerUtils...")
            tok_utils = TokenizerUtils(
                tokenizer          = tokenizer,
                max_input_length   = model_config.get("max_input_length",  512),
                max_output_length  = model_config.get("max_output_length", 128),
                lang_tags          = model_config.get("language_tags", INDICBART_LANG_TAGS),
            )
            logging.info(f"  Tokenizer vocab size: {tok_utils.get_num_added_tokens()}")

            # ----------------------------------------------------------
            # Step 4: Tokenize train + val DataFrames
            # ----------------------------------------------------------
            logging.info("\nStep 4/5: Tokenizing datasets...")
            logging.info("  Tokenizing train split...")
            train_dataset = tok_utils.tokenize_dataframe(
                train_df, mode="train", num_proc=1
            )

            logging.info("  Tokenizing val split...")
            val_dataset = tok_utils.tokenize_dataframe(
                val_df, mode="train", num_proc=1
            )

            logging.info(
                f"  Train dataset: {len(train_dataset)} examples, "
                f"columns: {train_dataset.column_names}"
            )
            logging.info(
                f"  Val dataset  : {len(val_dataset)} examples"
            )

            # ----------------------------------------------------------
            # Step 5: Initialize ModelTrainer and run training
            # ----------------------------------------------------------
            logging.info("\nStep 5/5: Starting Phase 1 training...")
            trainer = ModelTrainer(
                phase                = 1,
                config_path          = self.phase_config_path,
                model                = model,
                tokenizer            = tokenizer,
                tokenizer_vocab_size = tok_utils.get_num_added_tokens(),
            )

            train_summary = trainer.train(
                train_dataset          = train_dataset,
                eval_dataset           = val_dataset,
                resume_from_checkpoint = resume_from_checkpoint,
            )

            # ----------------------------------------------------------
            # Final summary
            # ----------------------------------------------------------
            duration_min = round((time.time() - start_time) / 60, 2)

            summary = {
                **train_summary,
                "languages"   : self.languages,
                "train_size"  : len(train_dataset),
                "val_size"    : len(val_dataset),
                "duration_min": duration_min,
            }

            logging.info("=" * 60)
            logging.info("PHASE 1 TRAINING PIPELINE COMPLETE")
            logging.info(f"  Duration   : {duration_min} minutes")
            logging.info(f"  Train loss : {summary.get('train_loss')}")
            logging.info(f"  Checkpoint : {CHECKPOINT_DIR}")
            logging.info("=" * 60)

            return summary

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase 1 monolingual ILSUM training for IndicBART"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Max rows per language (e.g. --max-samples 500 for a quick test run). "
            "Omit to use ALL data for full training."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint to resume training from. "
            "e.g. --resume artifacts/checkpoints/phase1/checkpoint-500"
        ),
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help=(
            "Languages to include. Default: Hindi Bengali English Gujarati. "
            "e.g. --languages Hindi Bengali"
        ),
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 1 TRAINING")
    print("=" * 60)
    if args.max_samples:
        print(f"⚠️  TEST MODE: max {args.max_samples} samples per language")
    else:
        print("🚀 FULL TRAINING MODE: using all available data")
    print()

    try:
        pipeline = Phase1TrainingPipeline(
            max_samples_per_language = args.max_samples,
            languages                = args.languages,
        )

        summary = pipeline.run(
            resume_from_checkpoint = args.resume
        )

        print("\n" + "=" * 60)
        print("✅ Phase 1 Training Complete!")
        print("=" * 60)
        print(f"  Duration    : {summary['duration_min']} minutes")
        print(f"  Train Loss  : {summary['train_loss']}")
        print(f"  Train Size  : {summary['train_size']} examples")
        print(f"  Val Size    : {summary['val_size']} examples")
        print(f"  Checkpoint  : {summary['output_dir']}")
        print(f"  Languages   : {summary['languages']}")
        print()
        print("Next step → run Phase 2: python -m src.pipelines.train_phase2")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
