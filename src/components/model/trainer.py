"""
Wraps HuggingFace Seq2SeqTrainer for IndicBART training.
Accepts a phase config YAML and runs the appropriate training phase
on the given dataset. Saves checkpoint to artifacts/checkpoints/phaseX/.
"""

import sys
import os
import yaml
import numpy as np
from typing import Optional, Dict, Tuple

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


class ModelTrainer:
    """
    Trains IndicBART for a specific training phase using HuggingFace
    Seq2SeqTrainer.

    Responsibilities:
        - Load a phase YAML config (phase1/2/3_train.yaml)
        - Build Seq2SeqTrainingArguments from config values
        - Resize model token embeddings if new language tags were added
        - Use DataCollatorForSeq2Seq for efficient dynamic padding
        - Run training + evaluation
        - Save final checkpoint to artifacts/checkpoints/phaseX/
        - Log training metrics and loss curves

    Supports all 3 training phases:
        Phase 1 — Monolingual ILSUM (teach summarization basics)
        Phase 2 — Synthetic + Real code-mixed (teach code-mixed handling)
        Phase 3 — Stabilization with ILSUM mix (prevent forgetting)

    DataCollatorForSeq2Seq vs fixed padding:
        We use DataCollatorForSeq2Seq instead of padding everything to
        max_length. It pads each batch only to the LONGEST sequence in
        that batch, which cuts memory usage dramatically when most articles
        are much shorter than 512 tokens — common in ILSUM summaries.
    """

    # Default output directory template (phaseX is substituted at runtime)
    CHECKPOINT_BASE = "artifacts/checkpoints"

    def __init__(
        self,
        phase: int,
        config_path: str,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        tokenizer_vocab_size: int,
    ):
        """
        Initialize ModelTrainer for a specific phase.

        Args:
            phase                : Training phase number (1, 2, or 3).
            config_path          : Path to the phase YAML config file.
                                   e.g. 'src/configs/phase1_train.yaml'
            model                : Loaded IndicBART model from ModelLoader.
            tokenizer            : IndicBART tokenizer from TokenizerUtils
                                   (after special tokens have been added).
            tokenizer_vocab_size : len(tokenizer) AFTER adding special tokens.
                                   Used to resize model embeddings.
        """
        try:
            self.phase                = phase
            self.config_path          = config_path
            self.model                = model
            self.tokenizer            = tokenizer
            self.tokenizer_vocab_size = tokenizer_vocab_size

            # Parse phase config
            self.config = self._load_phase_config()

            # Output dirs
            self.output_dir = os.path.join(
                self.CHECKPOINT_BASE, f"phase{self.phase}"
            )
            self.logging_dir = os.path.join(
                self.output_dir, "logs"
            )
            os.makedirs(self.output_dir,  exist_ok=True)
            os.makedirs(self.logging_dir, exist_ok=True)

            # Resize model embeddings to account for added language tokens
            self._resize_model_embeddings()

            logging.info(
                f"ModelTrainer initialized\n"
                f"  Phase       : {self.phase}\n"
                f"  Config      : {self.config_path}\n"
                f"  Output dir  : {self.output_dir}\n"
                f"  Epochs      : {self.config.get('epochs')}\n"
                f"  LR          : {self.config.get('learning_rate')}\n"
                f"  Batch size  : {self.config.get('batch_size')}\n"
                f"  Grad accum  : {self.config.get('gradient_accumulation_steps')}\n"
                f"  Mixed prec  : {self.config.get('mixed_precision')}"
            )

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Config loading                                                      #
    # ------------------------------------------------------------------ #

    def _load_phase_config(self) -> dict:
        """
        Load and return the phase training YAML as a dictionary.

        Returns:
            Dict with keys: epochs, learning_rate, batch_size,
                            gradient_accumulation_steps, mixed_precision,
                            and phase-specific keys (load_from, ratios, etc.)
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(
                    f"Phase config not found: '{self.config_path}'"
                )

            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            logging.info(f"Phase {self.phase} config loaded: {config}")
            return config

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Embedding resize                                                    #
    # ------------------------------------------------------------------ #

    def _resize_model_embeddings(self):
        """
        Resize the model's token embedding matrix to match the tokenizer
        vocabulary size after new language tags were added.

        Why this is critical:
            When we add <2hi>, <2bn>, <2gu>, <2en> via add_special_tokens(),
            the tokenizer vocabulary grows but the model's nn.Embedding layer
            still has the OLD number of rows. Trying to look up the new token
            IDs would cause an IndexError.

            resize_token_embeddings() adds new rows initialized with the MEAN
            of existing embeddings — a much better starting point than random
            initialization, since it stays in the same embedding space.
        """
        try:
            old_size = self.model.config.vocab_size
            self.model.resize_token_embeddings(self.tokenizer_vocab_size)
            new_size = self.model.config.vocab_size

            if new_size != old_size:
                logging.info(
                    f"Resized model embeddings: {old_size} → {new_size} tokens "
                    f"({new_size - old_size} new rows added)"
                )
            else:
                logging.info(
                    f"Model embeddings unchanged — vocab size: {new_size}"
                )

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Training arguments                                                  #
    # ------------------------------------------------------------------ #

    def _build_training_args(self) -> Seq2SeqTrainingArguments:
        """
        Build Seq2SeqTrainingArguments from the phase YAML config.

        Key arguments explained:
            predict_with_generate  — enables beam search during eval
                                     (required for ROUGE on real outputs)
            generation_max_length  — max tokens for eval summaries
            load_best_model_at_end — saves best checkpoint based on eval loss
            metric_for_best_model  — what 'best' means (eval_loss here)
            fp16                   — mixed precision (faster on GPU, safe on CPU off)
            dataloader_pin_memory  — speeds up GPU data transfer
            report_to='none'       — disable W&B/tensorboard for simplicity

        Returns:
            Configured Seq2SeqTrainingArguments object
        """
        try:
            epochs        = self.config.get("epochs", 3)
            lr            = self.config.get("learning_rate", 3e-5)
            batch_size    = self.config.get("batch_size", 2)
            grad_accum    = self.config.get("gradient_accumulation_steps", 8)
            mixed_prec    = self.config.get("mixed_precision", False)

            # Effective batch size = batch_size × grad_accum × num_gpus
            effective_batch = batch_size * grad_accum
            logging.info(
                f"Effective batch size: {batch_size} × {grad_accum} = {effective_batch}"
            )

            # Use fp16 ONLY if CUDA is available AND mixed_precision is True
            use_fp16 = mixed_prec and torch.cuda.is_available()
            if mixed_prec and not torch.cuda.is_available():
                logging.warning(
                    "mixed_precision=true in config but no CUDA GPU found. "
                    "Falling back to fp32 (CPU/MPS don't support fp16 training)."
                )

            args = Seq2SeqTrainingArguments(
                # Paths
                output_dir                  = self.output_dir,
                logging_dir                 = self.logging_dir,

                # Training schedule
                num_train_epochs            = epochs,
                learning_rate               = lr,
                per_device_train_batch_size = batch_size,
                per_device_eval_batch_size  = batch_size,
                gradient_accumulation_steps = grad_accum,

                # Precision
                fp16                        = use_fp16,

                # Evaluation strategy — eval every epoch
                eval_strategy               = "epoch",
                save_strategy               = "epoch",
                load_best_model_at_end      = True,
                metric_for_best_model       = "eval_loss",
                greater_is_better           = False,

                # Generation (for predict_with_generate during eval)
                predict_with_generate       = True,
                generation_max_length       = 128,
                generation_num_beams        = 4,

                # Save only best + last checkpoint to save disk space
                save_total_limit            = 2,

                # Logging
                logging_steps               = 50,
                logging_strategy            = "steps",

                # Misc
                warmup_steps                = 100,
                weight_decay                = 0.01,
                dataloader_pin_memory       = torch.cuda.is_available(),
                report_to                   = "none",   # No W&B/TB
                seed                        = 42,
            )

            logging.info(f"Seq2SeqTrainingArguments built for Phase {self.phase}")
            return args

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Data collator                                                       #
    # ------------------------------------------------------------------ #

    def _build_data_collator(self) -> DataCollatorForSeq2Seq:
        """
        Build DataCollatorForSeq2Seq for dynamic per-batch padding.

        Unlike padding everything to 512 upfront, this collator pads each
        mini-batch only to its longest sequence. On ILSUM where most
        articles are 100–300 tokens, this reduces compute per step
        significantly compared to always processing 512 tokens.

        label_pad_token_id = -100 ensures padded label positions are
        excluded from the loss calculation (consistent with TokenizerUtils).

        Returns:
            Configured DataCollatorForSeq2Seq
        """
        try:
            collator = DataCollatorForSeq2Seq(
                tokenizer         = self.tokenizer,
                model             = self.model,
                label_pad_token_id= -100,
                pad_to_multiple_of= 8 if torch.cuda.is_available() else None,
                # pad_to_multiple_of=8 aligns tensor dimensions for Tensor Core
                # hardware acceleration on NVIDIA GPUs (minor speedup)
            )
            logging.info("DataCollatorForSeq2Seq initialized")
            return collator

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Metrics (ROUGE for seq2seq evaluation)                             #
    # ------------------------------------------------------------------ #

    def _build_compute_metrics(self):
        """
        Build the compute_metrics function for Seq2SeqTrainer.

        Computes ROUGE-1, ROUGE-2, ROUGE-L during eval steps when
        predict_with_generate=True. This gives more meaningful signals
        than loss alone (loss can decrease while actual summary quality
        stays flat or degrades).

        The function is a closure that captures self.tokenizer so it can
        decode generated IDs without needing it as an argument.

        Returns:
            Callable that takes EvalPrediction and returns Dict[str, float]
        """
        try:
            from rouge_score import rouge_scorer as rouge_lib

            scorer = rouge_lib.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=False
            )

            def compute_metrics(eval_pred):
                """
                Decode generated IDs and reference labels, compute ROUGE.

                eval_pred.predictions : np.ndarray of generated token IDs
                eval_pred.label_ids   : np.ndarray of reference token IDs
                """
                predictions, label_ids = eval_pred

                # Replace -100 in labels (can't decode -100)
                label_ids = np.where(
                    label_ids != -100,
                    label_ids,
                    self.tokenizer.pad_token_id
                )

                # Decode predictions and references
                decoded_preds = self.tokenizer.batch_decode(
                    predictions, skip_special_tokens=True
                )
                decoded_labels = self.tokenizer.batch_decode(
                    label_ids, skip_special_tokens=True
                )

                # Strip whitespace
                decoded_preds  = [p.strip() for p in decoded_preds]
                decoded_labels = [l.strip() for l in decoded_labels]

                # Accumulate ROUGE scores
                r1_scores, r2_scores, rl_scores = [], [], []
                for pred, ref in zip(decoded_preds, decoded_labels):
                    if not ref:   # Skip empty references
                        continue
                    scores = scorer.score(ref, pred)
                    r1_scores.append(scores["rouge1"].fmeasure)
                    r2_scores.append(scores["rouge2"].fmeasure)
                    rl_scores.append(scores["rougeL"].fmeasure)

                result = {
                    "rouge1": round(np.mean(r1_scores) * 100, 2) if r1_scores else 0.0,
                    "rouge2": round(np.mean(r2_scores) * 100, 2) if r2_scores else 0.0,
                    "rougeL": round(np.mean(rl_scores) * 100, 2) if rl_scores else 0.0,
                }
                logging.info(f"Eval ROUGE: {result}")
                return result

            return compute_metrics

        except ImportError:
            logging.warning(
                "rouge_score not installed — ROUGE metrics disabled during eval. "
                "Training will continue using eval_loss only. "
                "Run: pip install rouge-score"
            )
            return None

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Public training interface                                           #
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict:
        """
        Run training for this phase.

        Steps:
            1. Build training arguments from phase YAML config
            2. Build DataCollatorForSeq2Seq (dynamic padding)
            3. Build ROUGE compute_metrics function
            4. Initialize Seq2SeqTrainer
            5. Run trainer.train()
            6. Save final model + tokenizer to checkpoint dir
            7. Return training metrics summary

        Args:
            train_dataset          : Tokenized HuggingFace Dataset for training.
                                     Produced by TokenizerUtils.tokenize_dataframe().
            eval_dataset           : Tokenized HuggingFace Dataset for validation.
            resume_from_checkpoint : Optional path to resume training from a
                                     previous checkpoint. Used when training is
                                     interrupted and needs to continue.
                                     e.g. 'artifacts/checkpoints/phase1/checkpoint-500'

        Returns:
            Dict with training metrics:
                {
                    'phase'        : int,
                    'output_dir'   : str,
                    'train_loss'   : float,
                    'eval_loss'    : float (last eval),
                    'epochs'       : int,
                }
        """
        try:
            logging.info("=" * 50)
            logging.info(f"Starting Phase {self.phase} Training")
            logging.info("=" * 50)
            logging.info(
                f"  Train examples : {len(train_dataset)}\n"
                f"  Eval examples  : {len(eval_dataset)}\n"
                f"  Output dir     : {self.output_dir}"
            )

            # Step 1: Build training args
            training_args = self._build_training_args()

            # Step 2: Build data collator
            data_collator = self._build_data_collator()

            # Step 3: Build ROUGE metrics (graceful fallback if not installed)
            compute_metrics = self._build_compute_metrics()

            # Step 4: Initialize Seq2SeqTrainer
            trainer = Seq2SeqTrainer(
                model           = self.model,
                args            = training_args,
                train_dataset   = train_dataset,
                eval_dataset    = eval_dataset,
                tokenizer       = self.tokenizer,
                data_collator   = data_collator,
                compute_metrics = compute_metrics,
            )

            logging.info(f"Seq2SeqTrainer initialized for Phase {self.phase}")

            # Step 5: Run training
            logging.info("Training started...")
            train_result = trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )

            # Step 6: Save final model + tokenizer
            logging.info(f"Saving final model to: {self.output_dir}")
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

            # Log training metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            # Step 7: Build and return summary
            summary = {
                "phase"      : self.phase,
                "output_dir" : self.output_dir,
                "train_loss" : round(metrics.get("train_loss", 0.0), 4),
                "epochs"     : self.config.get("epochs"),
            }

            logging.info("=" * 50)
            logging.info(f"Phase {self.phase} Training Complete!")
            logging.info(f"  Train loss : {summary['train_loss']}")
            logging.info(f"  Checkpoint : {self.output_dir}")
            logging.info("=" * 50)

            return summary

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def evaluate(self, eval_dataset: Dataset) -> Dict:
        """
        Run standalone evaluation on a dataset (without training).
        Useful for evaluating a loaded checkpoint on the test set.

        Args:
            eval_dataset : Tokenized HuggingFace Dataset to evaluate on.

        Returns:
            Dict of evaluation metrics (eval_loss, rouge1, rouge2, rougeL)
        """
        try:
            logging.info(f"Running standalone evaluation — {len(eval_dataset)} examples")

            training_args   = self._build_training_args()
            data_collator   = self._build_data_collator()
            compute_metrics = self._build_compute_metrics()

            trainer = Seq2SeqTrainer(
                model           = self.model,
                args            = training_args,
                eval_dataset    = eval_dataset,
                tokenizer       = self.tokenizer,
                data_collator   = data_collator,
                compute_metrics = compute_metrics,
            )

            eval_metrics = trainer.evaluate()
            logging.info(f"Evaluation metrics: {eval_metrics}")
            return eval_metrics

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


# ------------------------------------------------------------------ #
#  Standalone runner (test this file individually)                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    try:
        import pandas as pd
        from src.components.model.model_loader import ModelLoader
        from src.components.model.tokenizer_utils import TokenizerUtils, INDICBART_LANG_TAGS

        print("=" * 50)
        print("Testing ModelTrainer with tiny dummy dataset...")
        print("=" * 50)

        # Step 1: Load model + tokenizer
        print("\nStep 1: Loading model...")
        loader = ModelLoader()
        model, tokenizer, device = loader.load()
        config = loader.get_config()

        # Step 2: Initialize TokenizerUtils
        print("Step 2: Initializing TokenizerUtils...")
        tok_utils = TokenizerUtils(
            tokenizer          = tokenizer,
            max_input_length   = config.get("max_input_length",  512),
            max_output_length  = config.get("max_output_length", 128),
            lang_tags          = config.get("language_tags", INDICBART_LANG_TAGS),
        )

        # Step 3: Create tiny dummy dataset (just to test Trainer initializes)
        print("Step 3: Creating tiny dummy dataset...")
        dummy_df = pd.DataFrame({
            "text"    : [
                "<2hi> सरकार ने नई government scheme शुरू की जिससे economy में सुधार होगा",
                "<2hi> भारत में education system में बड़े changes हो रहे हैं students के लिए",
                "<2hi> नई technology से agriculture में विकास हो रहा है farmers के लिए",
            ],
            "summary" : [
                "सरकार ने नई योजना शुरू की",
                "शिक्षा प्रणाली में सुधार",
                "तकनीक से कृषि में विकास",
            ],
            "language": ["Hindi", "Hindi", "Hindi"],
        })

        train_ds = tok_utils.tokenize_dataframe(dummy_df, mode="train")
        eval_ds  = tok_utils.tokenize_dataframe(dummy_df, mode="train")
        print(f"  Train dataset: {len(train_ds)} examples")
        print(f"  Eval dataset : {len(eval_ds)} examples")

        # Step 4: Initialize ModelTrainer (Phase 1)
        print("Step 4: Initializing ModelTrainer...")
        trainer = ModelTrainer(
            phase                = 1,
            config_path          = "src/configs/phase1_train.yaml",
            model                = model,
            tokenizer            = tokenizer,
            tokenizer_vocab_size = tok_utils.get_num_added_tokens(),
        )

        print("\n✅ ModelTrainer initialized successfully!")
        print(f"   Output dir : {trainer.output_dir}")
        print(f"   Config     : {trainer.config}")
        print("\n⚠️  Not running full training in test mode (would take hours).")
        print("   Call trainer.train(train_ds, eval_ds) to start actual training.")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
