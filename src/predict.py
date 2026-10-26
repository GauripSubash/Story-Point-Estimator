"""Load a trusted local pipeline for prediction without training."""
import argparse
import joblib
from src.common import ROOT, read_json, sha256
from src.data_rules import combine_text, text_key

RANGES = {"low": "greater than 0 and at most 2 points", "medium": "greater than 2 and at most 5 points", "high": "greater than 5 and at most 100 points (training scope)"}

def load_model(path=None):
    path = path or ROOT / "models/selected.joblib"
    if not path.exists():
        raise FileNotFoundError("No saved model found. Restore models/selected.joblib from this frozen experiment before predicting.")
    frozen = read_json(ROOT / "reports/frozen_experiment.json")
    if sha256(path) != frozen["model_sha256"]["selected.joblib"]:
        raise ValueError("Saved model differs from the frozen experiment.")
    # joblib is executable serialization: load only the local model you trained.
    return joblib.load(path)

def predict_ticket(title, description="", model=None):
    text = combine_text(title, description)
    if not text_key(text):
        raise ValueError("Enter a title or description with ticket text. Blank input and effort disclosures alone cannot be estimated.")
    model = model if model is not None else load_model()
    vector = model.named_steps["tfidf"].transform([text])
    if vector.nnz == 0:
        raise ValueError("No recognizable vocabulary found. Add a more detailed ticket title or description.")
    category = str(model.predict([text])[0])
    return {"category": category, "range": RANGES[category], "note": "Team-relative category; discuss with the team. This is not an hours estimate or a sprint commitment."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="")
    args = parser.parse_args()
    try:
        result = predict_ticket(args.title, args.description)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, str(error) + "\n")
    print(f"{result['category'].upper()}: {result['range']}\n{result['note']}")

if __name__ == "__main__":
    main()
