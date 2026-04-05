"""
Optional transliteration step using the `indic-transliteration` library.
Converts Romanized Indic words to native script (e.g., "aaj" → "आज").
"""

from indic_transliteration import sanscript
from indic_transliteration.detect import detect
from typing import Optional


class IndicTransliteration:
    """
    Transliterates Indic text between Roman (IAST/ITRANS) and native scripts.

    Usage:
        trans = IndicTransliteration()

        # Auto-detect and convert to Devanagari
        devanagari = trans.to_devanagari("aaj main jaa raha hoon")
        # Output: "आज मैं जा रहा हूँ"

        # Convert from Devanagari to Roman
        roman = trans.to_roman("नमस्ते दुनिया")
        # Output: "namaste duniya"

        # Convert between specific scripts
        bengali = trans.convert("namaste", sanscript.BENGALI)
        # Output: "নমস্তে"
    """

    def __init__(self, default_from_script: str = sanscript.ITRANS):
        """
        Initialize the transliterator.

        Args:
            default_from_script: Default source script for Roman → Native conversion
                                 (default: ITRANS, supports IAST, SLP1, etc.)
        """
        self.default_from_script = default_from_script

    def detect_script(self, text: str) -> str:
        """
        Detect the script of the given text.

        Args:
            text: Text to analyze

        Returns:
            Script code (e.g., 'itrans', 'devanagari', 'bengali')
        """
        return detect(text)

    def to_devanagari(self, text: str) -> str:
        """
        Convert Romanized Indic text to Devanagari script.

        Automatically detects the source script (ITRANS, IAST, SLP1, etc.).

        Args:
            text: Romanized Indic text (e.g., "namaste", "aaj main jaa raha hoon")

        Returns:
            Text in Devanagari script (e.g., "नमस्ते", "आज मैं जा रहा हूँ")
        """
        try:
            detected_script = detect(text)
            return sanscript.transliterate(text, detected_script, sanscript.DEVANAGARI)
        except Exception as e:
            print(f"Warning: Could not transliterate to Devanagari: {e}")
            return text

    def to_roman(self, text: str, from_script: Optional[str] = None) -> str:
        """
        Convert Indic text to Roman script (ITRANS).

        Args:
            text: Text in Indic script (Devanagari, Bengali, Tamil, etc.)
            from_script: Optional source script (default: auto-detect or ITRANS)

        Returns:
            Text in Roman script (ITRANS format)
        """
        try:
            if from_script is None:
                from_script = detect(text)
            return sanscript.transliterate(text, from_script, self.default_from_script)
        except Exception as e:
            print(f"Warning: Could not transliterate to Roman: {e}")
            return text

    def convert(self, text: str, to_script: str, from_script: Optional[str] = None) -> str:
        """
        Convert text from one script to another.

        Args:
            text: Input text
            to_script: Target script (e.g., sanscript.BENGALI, sanscript.TAMIL)
            from_script: Optional source script (default: auto-detect)

        Returns:
            Text in target script
        """
        try:
            if from_script is None:
                from_script = detect(text)
            return sanscript.transliterate(text, from_script, to_script)
        except Exception as e:
            print(f"Warning: Could not convert to {to_script}: {e}")
            return text

    def transliterate_dataset(self, dataset, text_column: str, summary_column: str = None):
        """
        Transliterate an entire dataset (e.g., Pandas DataFrame).

        Args:
            dataset: Dataset containing text (e.g., Pandas DataFrame, HuggingFace Dataset)
            text_column: Column name containing text to transliterate
            summary_column: Optional column name for summaries (if also needs transliteration)

        Returns:
            Transliterated dataset (same type as input)
        """
        # Handle Pandas DataFrame
        if hasattr(dataset, 'copy'):
            df = dataset.copy()
            df[text_column] = df[text_column].apply(self.to_devanagari)
            if summary_column:
                df[summary_column] = df[summary_column].apply(self.to_devanagari)
            return df

        # Handle HuggingFace Dataset
        elif hasattr(dataset, 'map'):
            def transliterate_fn(examples):
                examples[text_column] = [self.to_devanagari(text) for text in examples[text_column]]
                if summary_column:
                    examples[summary_column] = [self.to_devanagari(summary) for summary in examples[summary_column]]
                return examples

            return dataset.map(transliterate_fn)

        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}. Use Pandas DataFrame or HuggingFace Dataset.")


# Example usage
if __name__ == "__main__":
    trans = IndicTransliteration()

    # Test detection
    text1 = "aaj main jaa raha hoon"
    text2 = "नमस्ते दुनिया"
    text3 = "வணக்கம் உலகம்"

    print(f"Text: '{text1}'")
    print(f"Detected script: {trans.detect_script(text1)}\n")

    print(f"Text: '{text2}'")
    print(f"Detected script: {trans.detect_script(text2)}\n")

    print(f"Text: '{text3}'")
    print(f"Detected script: {trans.detect_script(text3)}\n")

    # Test Roman → Devanagari
    devanagari = trans.to_devanagari(text1)
    print(f"Roman → Devanagari:")
    print(f"  Input:  {text1}")
    print(f"  Output: {devanagari}\n")

    # Test Devanagari → Roman
    roman = trans.to_roman(text2)
    print(f"Devanagari → Roman:")
    print(f"  Input:  {text2}")
    print(f"  Output: {roman}\n")

    # Test specific script conversion
    bengali = trans.convert(text2, sanscript.BENGALI)
    print(f"Devanagari → Bengali:")
    print(f"  Input:  {text2}")
    print(f"  Output: {bengali}\n")

    # Test with different ITRANS variants
    text_iast = "kārya"
    text_slp1 = "kArya"

    print(f"Different ITRANS variants:")
    print(f"  IAST: {trans.to_devanagari(text_iast)}")
    print(f"  SLP1: {trans.to_dev
