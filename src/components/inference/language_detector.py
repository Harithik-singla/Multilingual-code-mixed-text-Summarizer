"""
Detects dominant language of input text to select the correct IndicBART language tag.
Uses Unicode script ranges — no external libraries needed.
"""

import unicodedata


# Unicode block ranges for each script
DEVANAGARI = (0x0900, 0x097F)   # Hindi
BENGALI     = (0x0980, 0x09FF)
GUJARATI    = (0x0A80, 0x0AFF)


def _count_script(text, start, end):
    return sum(1 for ch in text if start <= ord(ch) <= end)


def detect_language_tag(text):
    """
    Returns the IndicBART language tag for the dominant script in text.
    Falls back to <2hi> for Latin/Romanized (Hinglish) input.
    """
    scores = {
        "<2hi>": _count_script(text, *DEVANAGARI),
        "<2bn>": _count_script(text, *BENGALI),
        "<2gu>": _count_script(text, *GUJARATI),
    }

    best_tag   = max(scores, key=scores.get)
    best_score = scores[best_tag]

    # If no Indic script detected → assume Romanized Hinglish
    if best_score == 0:
        return "<2hi>"

    return best_tag


if __name__ == "__main__":
    samples = [
        ("हिंदी समाचार लेख", "<2hi>"),
        ("বাংলা খবর", "<2bn>"),
        ("ગુજરાતી સમાચાર", "<2gu>"),
        ("aaj ka mausam kaisa hai bhai", "<2hi>"),
    ]
    for text, expected in samples:
        result = detect_language_tag(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text[:30]}' → {result} (expected {expected})")
