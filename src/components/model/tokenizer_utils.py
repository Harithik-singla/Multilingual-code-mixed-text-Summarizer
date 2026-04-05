"""
Adds language tag tokens to tokenizer and converts DataFrames
into tokenized HuggingFace Datasets ready for training.
"""

import sys
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging

# Language tags used in this project
LANG_TAGS = ["<2hi>", "<2bn>", "<2gu>", "<2en>"]


class TokenizerUtils:
    def __init__(self, tokenizer, max_input_length=512, max_output_length=128):
        try:
            self.tokenizer         = tokenizer
            self.max_input_length  = max_input_length
            self.max_output_length = max_output_length

            # Add language tags as special tokens so they don't get split
            # e.g. <2hi> should be ONE token, not "<", "2", "hi", ">"
            new_tokens = [t for t in LANG_TAGS if t not in tokenizer.get_vocab()]
            if new_tokens:
                self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
                logging.info(f"Added {len(new_tokens)} language tokens: {new_tokens}")
            else:
                logging.info("Language tokens already in vocab, no new tokens added")

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def vocab_size(self):
        """Returns current tokenizer vocab size (after adding new tokens)."""
        return len(self.tokenizer)

    def tokenize_batch(self, batch):
        """
        Tokenizes a batch of (text, summary) pairs.
        Called internally by tokenize_dataframe via Dataset.map().
        
        Labels use -100 for padding so the model ignores pad tokens in loss.
        """
        try:
            # Tokenize input text
            inputs = self.tokenizer(
                batch["text"],
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
            )

            # Tokenize summary (target output)
            targets = self.tokenizer(
                batch["summary"],
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
            )

            # Replace pad token ids in labels with -100
            # This tells the model: "don't calculate loss on padding positions"
            labels = [
                [(token if token != self.tokenizer.pad_token_id else -100)
                 for token in label_row]
                for label_row in targets["input_ids"]
            ]

            inputs["labels"] = labels
            return inputs

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def tokenize_dataframe(self, df):
        """
        Takes a DataFrame with 'text' and 'summary' columns,
        returns a tokenized HuggingFace Dataset ready for Seq2SeqTrainer.
        """
        try:
            logging.info(f"Tokenizing {len(df)} rows...")

            # Convert DataFrame to HuggingFace Dataset
            dataset = Dataset.from_pandas(df.reset_index(drop=True))

            # Apply tokenization to every batch
            tokenized = dataset.map(
                self.tokenize_batch,
                batched=True,
                batch_size=256,
                remove_columns=dataset.column_names,  # keep only input_ids, attention_mask, labels
                desc="Tokenizing"
            )

            logging.info(f"Tokenization done — {len(tokenized)} examples")
            return tokenized

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def tokenize_for_inference(self, text):
        """
        Tokenizes a single input text string for inference (no summary needed).
        Returns PyTorch tensors for model.generate().
        """
        try:
            return self.tokenizer(
                text,
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def decode(self, generated_ids):
        """Decodes model output token IDs back to a readable string."""
        try:
            return self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).strip()
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    from src.components.model.model_loader import ModelLoader

    loader = ModelLoader()
    model, tokenizer, device = loader.load()

    tok = TokenizerUtils(tokenizer)

    # Quick test
    df = pd.DataFrame({
        "text"    : ["<2hi> सरकार ने नई government scheme शुरू की"],
        "summary" : ["सरकार ने नई योजना शुरू की"],
        "language": ["Hindi"]
    })

    dataset = tok.tokenize_dataframe(df)
    print("Columns:", dataset.column_names)
    print("Example input_ids[:5]:", dataset[0]["input_ids"][:5])
    print("Example labels[:5]:", dataset[0]["labels"][:5])
    print("Vocab size:", tok.vocab_size())
