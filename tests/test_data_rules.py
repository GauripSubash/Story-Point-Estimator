import unittest
import pandas as pd
from src.data_rules import label_category, combine_text, text_key, assert_split_isolation, UnionFind
from src.extract_data import parse_values

class DataRuleTests(unittest.TestCase):
    def test_boundaries_and_invalid_labels(self):
        for value in (None, "", "abc", -1, 0, float("nan"), float("inf")):
            self.assertIsNone(label_category(value))
        for value, expected in ((0.1, "low"), (2, "low"), (2.01, "medium"), (5, "medium"), (5.01, "high")):
            self.assertEqual(label_category(value), expected)

    def test_training_and_prediction_cleaning(self):
        self.assertEqual(combine_text("<b>Fix</b>", "API\nStory points: 8\nsee https://example.org"), "Fix API see URL")
        self.assertEqual(combine_text(None, ""), "")
        self.assertEqual(combine_text("5 story points", "SP=3"), "")
        self.assertEqual(text_key("Fix: API!"), text_key("fix api"))

    def test_mysql_escapes_and_quoted_delimiters(self):
        sql = r"(1,'NULL','comma, (paren); \'quote\' and \\path\nline',NULL),(2,'x','it''s fine',3.5);"
        rows = list(parse_values(sql))
        self.assertEqual(rows[0], ["1", "NULL", "comma, (paren); 'quote' and \\path\nline", None])
        self.assertEqual(rows[1], ["2", "x", "it's fine", "3.5"])
        with self.assertRaises(ValueError):
            list(parse_values("(NOW());"))

    def test_leakage_assertion_catches_each_identifier(self):
        base = {"ID": ["1", "2"], "Issue_Key": ["P-1", "P-2"], "text_hash": ["a", "b"], "duplicate_group": ["x", "y"], "split": ["train", "test"]}
        assert_split_isolation(pd.DataFrame(base))
        for key in ("ID", "Issue_Key", "text_hash", "duplicate_group"):
            frame = pd.DataFrame(base)
            frame.loc[1, key] = frame.loc[0, key]
            with self.assertRaises(AssertionError):
                assert_split_isolation(frame)

    def test_transitive_duplicate_links(self):
        groups = UnionFind()
        groups.union("3", "2")
        groups.union("2", "1")
        self.assertEqual(groups.find("3"), groups.find("1"))

if __name__ == "__main__":
    unittest.main()
