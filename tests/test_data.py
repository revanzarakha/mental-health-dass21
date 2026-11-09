from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import SCORE_COLUMNS, detect_question_columns, severity_label  # noqa: E402

DATA_PATH = ROOT / "data" / "dass21_deidentified.csv"


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(DATA_PATH)
        cls.question_columns = detect_question_columns(cls.df)

    def test_shape(self) -> None:
        self.assertEqual(self.df.shape, (443, 24))

    def test_twenty_one_items(self) -> None:
        self.assertEqual(len(self.question_columns), 21)
        self.assertEqual(set(self.df.columns), set(self.question_columns + SCORE_COLUMNS))

    def test_no_missing_values(self) -> None:
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_item_range(self) -> None:
        values = self.df[self.question_columns].stack()
        self.assertTrue(values.between(0, 3).all())

    def test_score_range(self) -> None:
        values = self.df[SCORE_COLUMNS].stack()
        self.assertTrue(values.between(0, 42).all())


class SeverityLabelTest(unittest.TestCase):
    def test_depression_boundaries(self) -> None:
        cases = {9: 0, 10: 1, 13: 1, 14: 2, 20: 2, 21: 3, 27: 3, 28: 4, 42: 4}
        for score, expected in cases.items():
            self.assertEqual(severity_label("depresi", score), expected, score)

    def test_anxiety_boundaries(self) -> None:
        cases = {7: 0, 8: 1, 9: 1, 10: 2, 14: 2, 15: 3, 19: 3, 20: 4, 42: 4}
        for score, expected in cases.items():
            self.assertEqual(severity_label("anxiety", score), expected, score)

    def test_stress_boundaries(self) -> None:
        cases = {14: 0, 15: 1, 18: 1, 19: 2, 25: 2, 26: 3, 33: 3, 34: 4, 42: 4}
        for score, expected in cases.items():
            self.assertEqual(severity_label("stress", score), expected, score)


class LabelCoverageTest(unittest.TestCase):
    def test_all_five_levels_exist(self) -> None:
        df = pd.read_csv(DATA_PATH)
        pairs = {
            "depresi": "Depresi_Score",
            "anxiety": "Kecemasan_Score",
            "stress": "Stress_Score",
        }
        for domain, column in pairs.items():
            labels = {severity_label(domain, score) for score in df[column]}
            self.assertEqual(labels, {0, 1, 2, 3, 4}, domain)


if __name__ == "__main__":
    unittest.main()
