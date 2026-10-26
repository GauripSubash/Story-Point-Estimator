"""One final evaluation of frozen family winners and the majority baseline."""
import argparse
import os
from datetime import datetime, timezone
import joblib
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "work/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from src.common import ROOT, read_json, sha256, write_json
from src.metrics import classification_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true", help="Check frozen artifacts and saved predictions without changing the experiment")
    args = parser.parse_args()
    frozen = read_json(ROOT / "reports/frozen_experiment.json")
    assert sha256(ROOT / "config.json") == frozen["config_sha256"], "Frozen config changed"
    assert sha256(ROOT / "reports/split_metadata.json") == frozen["split_metadata_sha256"]
    for name, expected in frozen["split_file_sha256"].items():
        assert sha256(ROOT / f"data/processed/{name}.csv") == expected, f"Changed {name} split"
    for name, expected in frozen["model_sha256"].items():
        assert sha256(ROOT / "models" / name) == expected, f"Changed model {name}"
    destination = ROOT / "reports/test_metrics.json"
    if destination.exists() and not args.verify_existing:
        raise SystemExit("Already evaluated. Use --verify-existing to verify this unchanged experiment.")
    test = pd.read_csv(ROOT / "data/processed/test.csv", keep_default_na=False)
    labels = frozen["configuration"]["labels"]
    predictions = test[["ID", "Issue_Key", "project", "label", "points"]].copy()
    results = {}
    for family in ("baseline", "logistic_regression", "multinomial_nb"):
        model = joblib.load(ROOT / f"models/{family}.joblib")
        inputs = np.zeros((len(test), 1)) if family == "baseline" else test.text
        pred = model.predict(inputs)
        predictions[family] = pred
        results[family] = classification_metrics(test.label, pred, labels)
        results[family]["per_project"] = {}
        for project, subset in test.groupby("project"):
            pm = classification_metrics(subset.label, pred[subset.index], labels)
            pm["classes_present"] = sorted(subset.label.unique().tolist())
            pm["small_sample"] = len(subset) < 30
            results[family]["per_project"][project] = pm
    if args.verify_existing:
        assert destination.exists(), "No existing results to verify"
        prior = read_json(destination)
        assert prior["models"] == results, "Recomputed metrics differ"
        stored = pd.read_csv(ROOT / "reports/test_predictions.csv", keep_default_na=False)
        pd.testing.assert_frame_equal(stored, predictions, check_dtype=False)
        print("Frozen hashes, reloaded predictions, and every saved metric match.")
        return
    selected = frozen["selected_family"]
    predictions.to_csv(ROOT / "reports/test_predictions.csv", index=False)
    errors = test.loc[predictions[selected].ne(test.label)].copy()
    errors["predicted"] = predictions.loc[errors.index, selected]
    sampled = errors.sample(min(12, len(errors)), random_state=frozen["configuration"]["seed"])
    sampled[["ID", "Issue_Key", "project", "points", "label", "predicted", "text"]].to_csv(ROOT / "reports/error_samples.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, family in zip(axes, results):
        ConfusionMatrixDisplay(np.array(results[family]["confusion_matrix"]), display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(family.replace("_", " "))
    fig.suptitle("Frozen test set: rows = actual category; columns = prediction")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/confusion_matrix.png", dpi=150)
    plt.close(fig)
    report = {"evaluated_utc": datetime.now(timezone.utc).isoformat(), "frozen_experiment_sha256": sha256(ROOT / "reports/frozen_experiment.json"), "selected_family": selected, "models": results, "selected_errors": len(errors), "sampled_errors": len(sampled), "test_class_distribution": {k: int(v) for k, v in test.label.value_counts().items()}, "interpretation": "Three-category prediction on later tickets; neither exact story-point accuracy nor proof of unseen-team generalization."}
    write_json(destination, report)
    for family, scores in results.items():
        print(f"{family}: test n={scores['test_size']}, accuracy={scores['accuracy']:.4f}, macro F1={scores['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
