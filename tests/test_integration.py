"""Run after training; exercise the actual model and Streamlit form."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from src.common import ROOT
from src.predict import load_model, predict_ticket

@unittest.skipUnless((ROOT / "models/selected.joblib").exists(), "Train model first")
class ModelTests(unittest.TestCase):
    def test_missing_and_changed_model(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "selected.joblib"
            with self.assertRaises(FileNotFoundError):
                load_model(path)
            path.write_bytes(b"not the frozen model")
            with self.assertRaisesRegex(ValueError, "frozen experiment"):
                load_model(path)

    def test_streamlit_missing_model(self):
        from streamlit.testing.v1 import AppTest
        with TemporaryDirectory() as directory, patch("src.common.ROOT", Path(directory)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("saved model is missing", app.error[0].value)
        self.assertEqual(len(app.button), 0)

    def test_blank_and_unknown_inputs(self):
        for title in ("", " \n\t", "Story points: 5", "!!!", "qzxqzxqzxqzx"):
            with self.assertRaises(ValueError):
                predict_ticket(title)

    def test_saved_pipeline_and_new_examples(self):
        first, second = load_model(), load_model()
        for title, description in (("Fix typo in documentation", "Update the installation instructions"), ("Add webhook retry handling", "Retry transient failures with exponential backoff"), ("Migrate storage engine", "Support existing clients and test migration rollback")):
            self.assertEqual(predict_ticket(title, description, first), predict_ticket(title, description, second))

    def test_streamlit_form_blank_and_valid(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.warning), 1)
        app.text_input[0].set_value("Add retry handling for webhook failures")
        app.text_area[0].set_value("Retry requests with exponential backoff and test timeout handling")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.subheader), 1)
        self.assertIn(app.subheader[0].value.lower(), ("low", "medium", "high"))
        self.assertIn("6,692", app.caption[-1].value)
        app.text_input[0].set_value("qzxqzxqzxqzx")
        app.text_area[0].set_value("")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("No recognizable vocabulary", app.warning[0].value)
        self.assertEqual(len(app.subheader), 0)

if __name__ == "__main__":
    unittest.main()
