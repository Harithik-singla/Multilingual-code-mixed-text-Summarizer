"""
Unit tests for preprocessor.py
"""

import unittest
from src.components.data.preprocessor import clean_text, preprocess_dataframe
import pandas as pd


class TestCleanText(unittest.TestCase):

    def test_removes_urls(self):
        result = clean_text("Check this https://example.com for info")
        self.assertNotIn("http", result)

    def test_removes_html(self):
        result = clean_text("Hello <b>world</b> &amp; more")
        self.assertNotIn("<b>", result)
        self.assertNotIn("&amp;", result)

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")

    def test_normalizes_whitespace(self):
        result = clean_text("too   many    spaces")
        self.assertEqual(result, "too many spaces")

    def test_adds_language_tag(self):
        df = pd.DataFrame({"text": ["नमस्ते"], "summary": ["hello"]})
        cleaned = preprocess_dataframe(df, "Hindi")
        self.assertTrue(cleaned["text"].iloc[0].startswith("<2hi>"))

    def test_drops_empty_rows(self):
        df = pd.DataFrame({"text": ["", "valid text"], "summary": ["", "summary"]})
        cleaned = preprocess_dataframe(df, "Hindi")
        self.assertEqual(len(cleaned), 1)


if __name__ == "__main__":
    unittest.main()
