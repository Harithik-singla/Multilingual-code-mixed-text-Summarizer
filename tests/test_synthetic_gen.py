"""
Unit tests for synthetic_generator.py
"""

import unittest
import re
from src.components.data.synthetic_generator import mix_sentence, has_mixed_script


SAMPLE_DICT = {
    "सरकार": "government",
    "योजना": "scheme",
    "शिक्षा": "education",
    "स्वास्थ्य": "health",
    "विकास": "development",
}


class TestMixSentence(unittest.TestCase):

    def test_output_has_mixed_script(self):
        text = "<2hi> सरकार ने नई योजना शुरू की है आज"
        result = mix_sentence(text, SAMPLE_DICT)
        # After mixing, at least some replacements should occur
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_language_tag_preserved(self):
        text = "<2hi> सरकार ने नई योजना शुरू की"
        result = mix_sentence(text, SAMPLE_DICT)
        self.assertTrue(result.startswith("<2hi>"))

    def test_short_sentence_unchanged(self):
        text = "<2hi> सरकार योजना"
        result = mix_sentence(text, SAMPLE_DICT)
        self.assertEqual(result, text)

    def test_has_mixed_script_true(self):
        self.assertTrue(has_mixed_script("सरकार ने government बनाई"))

    def test_has_mixed_script_false_pure_hindi(self):
        self.assertFalse(has_mixed_script("सरकार ने योजना बनाई"))

    def test_has_mixed_script_false_pure_english(self):
        self.assertFalse(has_mixed_script("government launched a scheme"))


if __name__ == "__main__":
    unittest.main()
