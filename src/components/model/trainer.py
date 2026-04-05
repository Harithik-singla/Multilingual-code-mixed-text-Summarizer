"""
Wraps HuggingFace Seq2SeqTrainer for IndicBART training.
Reads a phase config YAML and trains the model on the given datasets.
Saves checkpoint to artifacts/checkpoints/phaseX/.
"""

import sys
import os
import yaml
import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


class ModelTrainer:
    def __init__(self, phase, config_path, model, tokenizer, vocab_size):
        """
        phase       : 1, 2, or 3
        config_path : path to phase YAML config
        model       : loaded IndicBART model
        tokenizer   : loaded IndicBART tokenizer (after adding special tokens)
        vocab_size  : len(tokenizer) after adding language tags
        """
        try:
            self.phase       = phase
            self.model       = model
            self.tokenizer   = tokenizer
            self.output_dir  = f"artifacts/checkpoints/phase{phase}"

            os.makedirs(self.output_dir, exist_ok=True)

            # Load phase training config
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

            # IMPORTANT: resize model embeddings to match new tokenizer vocab
            # (needed because we added new language tag tokens)
            self.model.resize_token_embeddings(vocab_size)
            logging.info(f"Resized model embeddings to vocab size: {vocab_size}")

            logging.info(f"ModelTrainer ready — Phase {phase}, output: {self.output_dir}")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def compute_rouge(self, eval_pred):
        """
        Computes ROUGE scores during evaluation.
        Called automatically by Seq2SeqTrainer after each eval epoch.
        """
        try:
            from rouge_score import rouge_scorer as rouge_lib

            predictions, label_ids = eval_pred

            # Replace -100 back to pad_token_id before decoding
            label_ids = np.where(
                label_ids != -100,
                label_ids,
                self.tokenizer.pad_token_id
            )

            # Decode to text
            pred_texts  = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            label_texts = self.tokenizer.batch_decode(label_ids,   skip_special_tokens=True)

            scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

            r1, r2, rl = [], [], []
            for pred, ref in zip(pred_texts, label_texts):
                if not ref.strip():
                    continue
                scores = scorer.score(ref.strip(), pred.strip())
                r1.append(scores["rouge1"].fmeasure)
                r2.append(scores["rouge2"].fmeasure)
                rl.append(scores["rougeL"].fmeasure)

            return {
                "rouge1": round(np.mean(r1) * 100, 2) if r1 else 0.0,
                "rouge2": round(np.mean(r2) * 100, 2) if r2 else 0.0,
                "rougeL": round(np.mean(rl) * 100, 2) if rl else 0.0,
            }

        except ImportError:
            logging.warning("rouge_score not installed — skipping ROUGE metrics")
            return {}
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def train(self, train_dataset, eval_dataset):
        """
        Runs training for this phase.
        Saves final model + tokenizer to artifacts/checkpoints/phaseX/.
        Returns training loss.
        """
        try:
            epochs      = self.config.get("epochs", 3)
            lr          = self.config.get("learning_rate", 3e-5)
            batch_size  = self.config.get("batch_size", 2)
            grad_accum  = self.config.get("gradient_accumulation_steps", 8)
            mixed_prec  = self.config.get("mixed_precision", False)

            # Only use fp16 (mixed precision) if we actually have a CUDA GPU
            use_fp16 = mixed_prec and torch.cuda.is_available()

            logging.info(
                f"Starting Phase {self.phase} training — "
                f"epochs={epochs}, lr={lr}, batch={batch_size}, "
                f"grad_accum={grad_accum}, fp16={use_fp16}"
            )

            training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,

                num_train_epochs=epochs,
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,

                fp16=use_fp16,

                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                save_total_limit=2,           # keep only 2 best checkpoints

                predict_with_generate=True,   # needed for ROUGE evaluation
                generation_max_length=128,

                logging_steps=50,
                warmup_steps=100,
                weight_decay=0.01,

                report_to="none",             # disable W&B / TensorBoard
                seed=42,
            )

            # DataCollator pads each batch to its longest sequence
            # (better than always padding everything to 512)
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                label_pad_token_id=-100,
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_rouge,
            )

            result = trainer.train()

            # Save final model and tokenizer
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

            train_loss = round(result.metrics.get("train_loss", 0.0), 4)
            logging.info(f"Phase {self.phase} training done — loss: {train_loss}")
            logging.info(f"Checkpoint saved to: {self.output_dir}")

            return train_loss

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    print("Import ModelTrainer from src.components.model.trainer to use it.")
    print("Run training via: python -m src.pipelines.train_phase1")
