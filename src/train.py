"""Compare a small fixed grid on validation only; freeze before final testing."""
import platform
import warnings
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.common import ROOT, config, read_json, sha256, write_json
from src.metrics import classification_metrics

def main():
    if (ROOT / "reports/test_metrics.json").exists():
        raise SystemExit("Final test was already evaluated. Keep this experiment frozen; use a separate experiment for new training.")
    cfg = config()
    split_meta = read_json(ROOT / "reports/split_metadata.json")
    assert sha256(ROOT / "config.json") == split_meta["config_sha256"], "Config changed since preparation"
    data = ROOT / "data/processed"
    train = pd.read_csv(data / "train.csv", keep_default_na=False)
    validation = pd.read_csv(data / "validation.csv", keep_default_na=False)
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    vec_cfg = dict(cfg["tfidf"])
    vec_cfg["ngram_range"] = tuple(vec_cfg["ngram_range"])
    vectorizer = TfidfVectorizer(**vec_cfg)
    xtrain = vectorizer.fit_transform(train.text)
    xval = vectorizer.transform(validation.text)
    majority = DummyClassifier(strategy="most_frequent", random_state=cfg["seed"])
    majority.fit(np.zeros((len(train), 1)), train.label)
    joblib.dump(majority, model_dir / "baseline.joblib")
    rows = []
    winners = {}
    baseline = classification_metrics(validation.label, majority.predict(np.zeros((len(validation), 1))), cfg["labels"])
    with threadpool_limits(limits=2):
        for family, settings in (("logistic_regression", cfg["logistic_C"]), ("multinomial_nb", cfg["nb_alpha"])):
            best_key = None
            for value in settings:
                model = (LogisticRegression(C=value, max_iter=1000, solver="lbfgs", random_state=cfg["seed"]) if family == "logistic_regression" else MultinomialNB(alpha=value))
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(xtrain, train.label)
                vm = classification_metrics(validation.label, model.predict(xval), cfg["labels"])
                tm = classification_metrics(train.label, model.predict(xtrain), cfg["labels"])
                row = {"family": family, "parameter": {"C" if family == "logistic_regression" else "alpha": value}, "validation": vm, "training": tm}
                rows.append(row)
                print(f"{family} {value}: validation accuracy={vm['accuracy']:.4f}, macro F1={vm['macro_f1']:.4f}", flush=True)
                key = (vm["macro_f1"], vm["accuracy"])
                if best_key is None or key > best_key:
                    best_key = key
                    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", model)])
                    joblib.dump(pipeline, model_dir / f"{family}.joblib")
                    winners[family] = row
    selected = max(winners, key=lambda family: (winners[family]["validation"]["macro_f1"], winners[family]["validation"]["accuracy"]))
    selected_pipeline = joblib.load(model_dir / f"{selected}.joblib")
    joblib.dump(selected_pipeline, model_dir / "selected.joblib")
    before = selected_pipeline.predict(validation.text.iloc[:50])
    after = joblib.load(model_dir / "selected.joblib").predict(validation.text.iloc[:50])
    assert np.array_equal(before, after), "Reload changed predictions"
    report = {"baseline": baseline, "candidates": rows, "family_winners": winners, "selected_family": selected, "selection": "highest validation macro F1, accuracy breaks ties; grid order breaks exact ties", "vocabulary_size": len(vectorizer.vocabulary_), "matrix_shape": list(xtrain.shape), "refit_on_train_plus_validation": False}
    write_json(ROOT / "reports/validation_metrics.json", report)
    frozen = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "configuration": cfg,
        "config_sha256": sha256(ROOT / "config.json"), "data_sha256": sha256(data / "tickets.csv"),
        "split_metadata_sha256": sha256(ROOT / "reports/split_metadata.json"),
        "split_file_sha256": {name: sha256(data / f"{name}.csv") for name in ("train", "validation", "test")},
        "model_sha256": {path.name: sha256(path) for path in sorted(model_dir.glob("*.joblib"))},
        "selected_family": selected, "parameters": {family: row["parameter"] for family, row in winners.items()},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "reload_predictions_match": True,
        "notes": "Fitted on train only; validation used for selection; no final test metrics observed before freeze.",
    }
    write_json(ROOT / "reports/frozen_experiment.json", frozen)
    print(f"Selected {selected}; experiment frozen; no test evaluation yet.")

if __name__ == "__main__":
    main()
