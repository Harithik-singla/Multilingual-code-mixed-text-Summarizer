"""
Integration test for predictor.py: verifies the full inference pipeline
runs without errors using the trained checkpoint.
"""

import unittest
import os


class TestPredictor(unittest.TestCase):

    @unittest.skipUnless(
        os.path.exists("artifacts/checkpoints/phase3/config.json") or
        os.path.exists("artifacts/checkpoints/phase2/config.json") or
        os.path.exists("artifacts/checkpoints/phase1/config.json"),
        "No checkpoint found — skipping predictor test"
    )
    def test_predict_hinglish(self):
        from src.components.inference.predictor import Predictor
        predictor = Predictor()
        summary = predictor.predict("aaj mausam bahut achha hai bhai")
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

    @unittest.skipUnless(
        os.path.exists("artifacts/checkpoints/phase3/config.json") or
        os.path.exists("artifacts/checkpoints/phase2/config.json") or
        os.path.exists("artifacts/checkpoints/phase1/config.json"),
        "No checkpoint found — skipping predictor test"
    )
    def test_predict_hindi(self):
        from src.components.inference.predictor import Predictor
        predictor = Predictor()
        summary = predictor.predict("भारत में आज बारिश हो रही है और मौसम ठंडा है")
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)


if __name__ == "__main__":
    unittest.main()
