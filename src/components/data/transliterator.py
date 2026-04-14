"""
Converts Romanized Bengali/Gujarati text to native script before inference.
Hindi/Hinglish is skipped — the model handles it natively via HinGE training.
"""

import sys
from indic_transliteration import sanscript

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


SCRIPT_MAP = {
    "<2bn>": sanscript.BENGALI,
    "<2gu>": sanscript.GUJARATI,
}


def transliterate_to_native(text, lang_tag):
    """
    Converts Romanized text to native script for Bengali/Gujarati.
    Returns text unchanged for Hindi/Hinglish/English.
    """
    try:
        if lang_tag not in SCRIPT_MAP:
            return text

        target_script = SCRIPT_MAP[lang_tag]
        result = sanscript.transliterate(text, sanscript.ITRANS, target_script)
        logging.info(f"Transliterated to {target_script}")
        return result

    except Exception as e:
        raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    test = "namaste duniya"
    print(f"Bengali:  {transliterate_to_native(test, '<2bn>')}")
    print(f"Gujarati: {transliterate_to_native(test, '<2gu>')}")
    print(f"Hindi:    {transliterate_to_native(test, '<2hi>')}")
