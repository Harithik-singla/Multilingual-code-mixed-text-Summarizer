"""
Tokenizer helper functions: adds special language tag tokens, tokenizes
input/summary columns, handles truncation, and prepares HuggingFace
Dataset objects ready for Seq2SeqTrainer.
"""

import sys
import os
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


# ------------------------------------------------------------------ #
#  IndicBART language tags that must be treated as single tokens      #
# ------------------------------------------------------------------ #

INDICBART_LANG_TAGS = ["<2hi>", "<2bn>", "<2gu>", "<2en>"]


class TokenizerUtils:
    """
    Prepares text data for IndicBART training and inference.

    Responsibilities:
        1. Add language tag tokens as special tokens to the tokenizer
           so they are NEVER split into sub-tokens by SentencePiece.
           e.g.  "<2hi>" must always be ONE token, not "<", "2", "hi", ">"

        2. Tokenize a DataFrame (text + summary columns) into a
           HuggingFace Dataset with:
               input_ids        — tokenized + truncated text
               attention_mask   — 1 for real tokens, 0 for padding
               labels           — tokenized summary with -100 for pad positions
                                  (Seq2SeqTrainer ignores -100 in loss calc)

        3. Support inference mode — tokenize text only (no summary column)

    IndicBART specifics:
        - Max input  length : 512 tokens (from model_config.yaml)
        - Max output length : 128 tokens (from model_config.yaml)
        - Padding strategy  : 'max_length' for uniform batch shapes
        - Truncation        : True for both input and labels
        - Label padding     : replace tokenizer.pad_token_id → -100

    Why -100 for label padding?
        PyTorch CrossEntropyLoss ignores positions where label == -100.
        This means the model is not penalized for padding positions in
        the summary, which would otherwise corrupt the loss signal.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_input_length: int  = 512,
        max_output_length: int = 128,
        lang_tags: Optional[List[str]] = None,
    ):
        """
        Initialize TokenizerUtils.

        Args:
            tokenizer         : Loaded IndicBART AutoTokenizer from ModelLoader.
            max_input_length  : Max tokens for input text. Default 512.
            max_output_length : Max tokens for summary/labels. Default 128.
            lang_tags         : Language tags to register as special tokens.
                                Defaults to INDICBART_LANG_TAGS.
        """
        try:
            self.tokenizer         = tokenizer
            self.max_input_length  = max_input_length
            self.max_output_length = max_output_length
            self.lang_tags         = lang_tags or INDICBART_LANG_TAGS

            # Register language tags as special tokens
            self._add_language_tokens()

            logging.info(
                f"TokenizerUtils initialized\n"
                f"  max_input_length  : {self.max_input_length}\n"
                f"  max_output_length : {self.max_output_length}\n"
                f"  lang_tags         : {self.lang_tags}\n"
                f"  vocab_size (after adding tokens): {len(self.tokenizer)}"
            )

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Special token registration                                          #
    # ------------------------------------------------------------------ #

    def _add_language_tokens(self) -> int:
        """
        Add IndicBART language tags as special tokens to the tokenizer.

        Why this is necessary:
            SentencePiece tokenizes "<2hi>" as ["<", "▁2", "hi", ">"] by default.
            This breaks the language conditioning: the model never sees a single
            <2hi> token at position 0; it sees 4 meaningless sub-tokens instead.

            Adding them as additional_special_tokens ensures SentencePiece
            treats each tag as a single, atomic token with its own embedding.

        Only adds tags that are NOT already in the vocabulary to avoid
        creating duplicate embeddings.

        Returns:
            Number of new tokens actually added (0 if all already existed)
        """
        try:
            # Find which tags are genuinely new (not already in vocab)
            new_tokens = [
                tag for tag in self.lang_tags
                if tag not in self.tokenizer.get_vocab()
            ]

            if not new_tokens:
                logging.info(
                    "All language tags already in tokenizer vocabulary — "
                    "no new tokens added."
                )
                return 0

            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )

            logging.info(
                f"Added {num_added} new language tag token(s) to tokenizer: "
                f"{new_tokens}"
            )
            return num_added

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def get_num_added_tokens(self) -> int:
        """
        Return total number of tokens added to the tokenizer.
        Used by the trainer to resize model embeddings after adding tokens.

        The trainer MUST call model.resize_token_embeddings(len(tokenizer))
        after TokenizerUtils is initialized, otherwise the new tag tokens
        will have no corresponding embedding rows in the model.

        Returns:
            Current total tokenizer vocab size (including added tokens)
        """
        return len(self.tokenizer)

    # ------------------------------------------------------------------ #
    #  Single-example tokenization                                         #
    # ------------------------------------------------------------------ #

    def _tokenize_example(
        self,
        text: str,
        summary: Optional[str] = None,
    ) -> Dict:
        """
        Tokenize a single (text, summary) pair.

        Input text is already pre-tagged: "<2hi> आज मौसम अच्छा है"
        Summary is clean, untagged: "आज मौसम अच्छा है"

        Processing:
            1. Tokenize text    → input_ids + attention_mask (max 512)
            2. Tokenize summary → label_ids                 (max 128)
            3. Replace pad token IDs in labels with -100

        Args:
            text    : Pre-cleaned, pre-tagged input string.
            summary : Clean summary string. None in inference mode.

        Returns:
            Dict with keys: input_ids, attention_mask, (labels if summary given)
        """
        try:
            # Tokenize input text (truncate to max_input_length)
            model_inputs = self.tokenizer(
                text,
                max_length  = self.max_input_length,
                padding     = "max_length",
                truncation  = True,
                return_tensors = None,  # Return plain lists (for Dataset.map)
            )

            # Tokenize summary/labels if provided (training mode)
            if summary is not None:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        summary,
                        max_length  = self.max_output_length,
                        padding     = "max_length",
                        truncation  = True,
                        return_tensors = None,
                    )

                # Replace pad token id with -100 so loss ignores them
                label_ids = [
                    (label_id if label_id != self.tokenizer.pad_token_id else -100)
                    for label_id in labels["input_ids"]
                ]
                model_inputs["labels"] = label_ids

            return model_inputs

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Batch tokenization (via HuggingFace Dataset.map)                   #
    # ------------------------------------------------------------------ #

    def _batch_tokenize(
        self,
        batch: Dict,
        mode: str = "train",
    ) -> Dict:
        """
        Batch tokenization function — passed to Dataset.map().

        Processes a batch of examples at once (more efficient than
        row-by-row processing for large datasets).

        Args:
            batch : Dict with keys 'text' (and 'summary' in train mode)
            mode  : 'train' → tokenize text + summary
                    'inference' → tokenize text only

        Returns:
            Dict with keys: input_ids, attention_mask, (labels if train mode)
        """
        try:
            # Tokenize all texts in batch
            model_inputs = self.tokenizer(
                batch["text"],
                max_length     = self.max_input_length,
                padding        = "max_length",
                truncation     = True,
                return_tensors = None,
            )

            # Tokenize summaries if training
            if mode == "train" and "summary" in batch:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        batch["summary"],
                        max_length     = self.max_output_length,
                        padding        = "max_length",
                        truncation     = True,
                        return_tensors = None,
                    )

                # Replace pad token ids with -100
                model_inputs["labels"] = [
                    [
                        (label_id if label_id != self.tokenizer.pad_token_id else -100)
                        for label_id in label_ids
                    ]
                    for label_ids in labels["input_ids"]
                ]

            return model_inputs

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  DataFrame → HuggingFace Dataset                                     #
    # ------------------------------------------------------------------ #

    def tokenize_dataframe(
        self,
        df: pd.DataFrame,
        mode: str         = "train",
        batch_size: int   = 256,
        num_proc: int     = 1,
    ) -> Dataset:
        """
        Convert a preprocessed DataFrame into a tokenized HuggingFace Dataset.

        Input DataFrame must have:
            text     : Pre-cleaned + pre-tagged text  (e.g. "<2hi> आज मौसम अच्छा है")
            summary  : Clean summary (required in train mode, ignored in inference mode)
            language : Language name (kept for reference, not used in tokenization)

        Output Dataset has columns:
            input_ids      : List[int] — token IDs for input text
            attention_mask : List[int] — 1 for real tokens, 0 for padding
            labels         : List[int] — token IDs for summary (-100 for padding)

        Args:
            df         : Preprocessed DataFrame from preprocessor.py output
            mode       : 'train' or 'inference'
            batch_size : Number of examples per batch in Dataset.map()
            num_proc   : Parallel workers for Dataset.map() (keep at 1 on Windows)

        Returns:
            HuggingFace Dataset ready for Seq2SeqTrainer
        """
        try:
            if mode not in ("train", "inference"):
                raise ValueError(
                    f"Invalid mode '{mode}'. Choose 'train' or 'inference'."
                )

            required_cols = ["text"]
            if mode == "train":
                required_cols.append("summary")

            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(
                        f"Column '{col}' missing from DataFrame. "
                        f"Available: {list(df.columns)}"
                    )

            logging.info(
                f"Tokenizing DataFrame — {len(df)} rows, mode='{mode}'"
            )

            # Convert pandas DataFrame to HuggingFace Dataset
            # (more memory-efficient than processing raw Python lists)
            hf_dataset = Dataset.from_pandas(df.reset_index(drop=True))

            # Apply batch tokenization
            tokenized = hf_dataset.map(
                lambda batch: self._batch_tokenize(batch, mode=mode),
                batched       = True,
                batch_size    = batch_size,
                num_proc      = num_proc,
                remove_columns= hf_dataset.column_names,  # Drop original text cols
                desc          = f"Tokenizing ({mode})",
            )

            logging.info(
                f"Tokenization complete — {len(tokenized)} examples\n"
                f"  Columns: {tokenized.column_names}"
            )
            return tokenized

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def tokenize_csv(
        self,
        csv_path: str,
        mode: str         = "train",
        batch_size: int   = 256,
        num_proc: int     = 1,
    ) -> Dataset:
        """
        Load a preprocessed CSV file and tokenize it into a HuggingFace Dataset.
        Convenience wrapper around tokenize_dataframe() for pipeline use.

        Args:
            csv_path   : Path to a *_clean.csv or *_synthetic.csv file
            mode       : 'train' or 'inference'
            batch_size : Batch size for Dataset.map()
            num_proc   : Parallel workers (keep at 1 on Windows)

        Returns:
            Tokenized HuggingFace Dataset
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"CSV file not found: '{csv_path}'"
                )

            logging.info(f"Loading CSV for tokenization: {csv_path}")
            df = pd.read_csv(csv_path)
            logging.info(f"  Loaded {len(df)} rows from {csv_path}")

            return self.tokenize_dataframe(df, mode=mode, batch_size=batch_size, num_proc=num_proc)

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Single-text inference tokenization                                  #
    # ------------------------------------------------------------------ #

    def tokenize_for_inference(self, text: str) -> Dict:
        """
        Tokenize a single raw input text for inference.
        Returns PyTorch tensors (batched with batch_size=1).

        The input text should already have a language tag prepended.
        If not, the caller is responsible for adding it.

        Args:
            text : Pre-tagged input text e.g. "<2hi> आज का मौसम बड़ा अच्छा है"

        Returns:
            Dict with 'input_ids' and 'attention_mask' as PyTorch tensors
            (batch dimension: 1 × seq_len)
        """
        try:
            logging.info(f"Tokenizing single input for inference: {text[:80]}...")

            inputs = self.tokenizer(
                text,
                max_length     = self.max_input_length,
                padding        = "max_length",
                truncation     = True,
                return_tensors = "pt",  # PyTorch tensors for model.generate()
            )

            logging.info(
                f"  Input token count: {inputs['input_ids'].shape[1]}"
            )
            return inputs

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def decode_output(self, generated_ids) -> str:
        """
        Decode model output token IDs back into a human-readable string.
        Skips special tokens (EOS, BOS, pad, language tags).

        Args:
            generated_ids : Token ID tensor from model.generate()
                            Shape: (batch_size, seq_len) or (seq_len,)

        Returns:
            Decoded summary string
        """
        try:
            # Handle both batched and single outputs
            if generated_ids.dim() == 2:
                # Take first item from batch
                ids = generated_ids[0]
            else:
                ids = generated_ids

            decoded = self.tokenizer.decode(
                ids,
                skip_special_tokens    = True,
                clean_up_tokenization_spaces = True,
            )

            return decoded.strip()

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


# ------------------------------------------------------------------ #
#  Standalone runner (test this file individually)                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    try:
        from src.components.model.model_loader import ModelLoader

        print("=" * 50)
        print("Testing TokenizerUtils...")
        print("=" * 50)

        # Step 1: Load tokenizer
        print("\nStep 1: Loading tokenizer via ModelLoader...")
        loader             = ModelLoader()
        model, tokenizer, device = loader.load()
        config             = loader.get_config()

        # Step 2: Initialize TokenizerUtils
        print("\nStep 2: Initializing TokenizerUtils...")
        tok_utils = TokenizerUtils(
            tokenizer          = tokenizer,
            max_input_length   = config.get("max_input_length",  512),
            max_output_length  = config.get("max_output_length", 128),
            lang_tags          = config.get("language_tags", INDICBART_LANG_TAGS),
        )

        # Step 3: Test single example tokenization
        print("\nStep 3: Testing single example tokenization...")
        test_text    = "<2hi> सरकार ने नई government scheme शुरू की"
        test_summary = "सरकार ने नई योजना शुरू की"

        result = tok_utils._tokenize_example(test_text, test_summary)
        print(f"  Input text   : {test_text}")
        print(f"  Summary      : {test_summary}")
        print(f"  input_ids    : {result['input_ids'][:10]}... (len={len(result['input_ids'])})")
        print(f"  attention_mask: {result['attention_mask'][:10]}...")
        print(f"  labels       : {result['labels'][:10]}... (len={len(result['labels'])})")

        # Step 4: Test DataFrame tokenization on a tiny sample
        print("\nStep 4: Testing DataFrame tokenization on tiny sample...")
        sample_df = pd.DataFrame({
            "text"    : [test_text, "<2bn> সরকার বাজার নীতি government করেছে"],
            "summary" : [test_summary, "সরকার নতুন নীতি করেছে"],
            "language": ["Hindi", "Bengali"],
        })

        tokenized_ds = tok_utils.tokenize_dataframe(sample_df, mode="train")
        print(f"  Dataset columns : {tokenized_ds.column_names}")
        print(f"  Dataset size    : {len(tokenized_ds)}")
        print(f"  input_ids shape : {len(tokenized_ds[0]['input_ids'])}")
        print(f"  labels shape    : {len(tokenized_ds[0]['labels'])}")

        # Step 5: Verify -100 masking in labels
        first_labels_sample = tokenized_ds[0]['labels']
        neg100_count = sum(1 for l in first_labels_sample if l == -100)
        print(f"\nStep 5: -100 masking in labels...")
        print(f"  Total label positions : {len(first_labels_sample)}")
        print(f"  Positions masked (-100): {neg100_count}")
        print(f"  Real label tokens     : {len(first_labels_sample) - neg100_count}")

        # Step 6: Test inference tokenization
        print("\nStep 6: Testing inference tokenization...")
        inference_input = tok_utils.tokenize_for_inference(test_text)
        print(f"  input_ids shape  : {inference_input['input_ids'].shape}")
        print(f"  attention_mask shape: {inference_input['attention_mask'].shape}")

        print("\n✅ TokenizerUtils test complete!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
