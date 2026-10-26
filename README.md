# Story Point Estimator

A local Python research demo that predicts **low, medium or high effort categories** from an issue title and description using TF-IDF text features. Compares logistic regression and Multinomial Naive Bayes with an always-low baseline; the demo uses the Naive Bayes model selected on validation.

## What the results support

Audited 458,232 unique TAWOS issues across 39 projects. Eligibility filtering and deduplication left 44,609 modeling tickets, split chronologically into 31,226 training, 6,691 validation and 6,692 test tickets.

Final test results:

- Always-low baseline: **62.49% accuracy**, macro F1 **0.2564**.
- Logistic regression (C=2): **61.76% accuracy**, macro F1 **0.4962**.
- Selected Naive Bayes (alpha=0.5): **65.60% accuracy**, macro F1 **0.5034**.

Naive Bayes correctly classified 4,390 test tickets, 208 more than the baseline: a 3.11 percentage-point accuracy improvement. No statistical significance claim is made. Medium recall is only 21.36%; high recall is 38.75%. This is a modest text-classification baseline with substantial limitations, not a reliable automatic estimator.

Categories are low (0,2], medium (2,5], high (5,100]. Story points are team-relative, not hours. Predicting a category is not predicting exact points. A large multi-service platform migration can still receive an implausible medium prediction; the model does not calculate dependencies or add up acceptance criteria. Split such initiatives into estimable work and use human judgment.

## Run the existing local experiment

Run commands in PowerShell from the project root. This machine uses `.venv312`, Python 3.12.14:

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
.\.venv312\Scripts\python.exe -m src.evaluate --verify-existing
.\.venv312\Scripts\python.exe -m src.predict --title "Fix typo in documentation" --description "Update the installation instructions"
.\.venv312\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501. Stop the server with Ctrl+C in its terminal. If port 8501 is already in use by this demo, use the existing server or select another port. The app loads the saved trusted local model without training; it rejects blank, disclosure-only and unknown-vocabulary inputs. No calibrated confidence percentage is shown.

The existing experiment needs ignored `data/processed/`, `models/` and detailed report files. Git alone does not preserve these. Restore a trusted copy of the frozen model if missing; do not load untrusted joblib files. `--verify-existing` requires saved test metrics and predictions and checks frozen hashes, predictions and every metric. It does not overwrite results or tune models.

## Setup on another machine

Use a standard Python 3.12 installation and a new environment in a **separate project directory**:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements.txt` lists seven pinned direct dependencies; `requirements-lock.txt` pins the 46 reachable installed packages. The fresh-machine installation has not been tested. On this machine `.venv312` uses `--system-site-packages` with bundled pandas/numpy and is not fully isolated or portable. Default Python 3.14 and a newer SciPy build encountered Windows Application Control DLL restrictions; the working versions are recorded in the lock and frozen experiment. Do not disable security controls. Sandbox temporary-file/socket restrictions required approved execution outside the sandbox for AppTest and the local server.

## Full pipeline for a fresh experiment only

These commands describe the implemented pipeline. **Do not run acquisition/extraction/audit over the current frozen workspace.** Use a separate checkout/directory with no existing generated artifacts; retain a copy of the original experiment evidence. Acquisition requires network access, a 637.6 MB archive download and disk space for extracted CSVs. Extraction reads a 4.31 GB uncompressed SQL stream without executing SQL or requiring MySQL.

```powershell
.\.venv312\Scripts\python.exe -m src.acquire_data
.\.venv312\Scripts\python.exe -m src.extract_data
.\.venv312\Scripts\python.exe -m src.audit_data
.\.venv312\Scripts\python.exe -m src.prepare_data
.\.venv312\Scripts\python.exe -m src.train
.\.venv312\Scripts\python.exe -m src.evaluate
.\.venv312\Scripts\python.exe -m src.evaluate --verify-existing
```

Preparation and training intentionally refuse to run when `reports/test_metrics.json` exists. Evaluation also refuses repeat writes. Do not delete final evidence to bypass these guards. Future changes require a separate experiment; this test set has already been inspected and is no longer an untouched evaluation set for iterative tuning.

The complete acquisition-to-training pipeline ran during checkpoints 1–3; checkpoint 6 verified saved artifacts, tests, inference and freeze guards rather than rebuilding over the final evidence. Identical model binary hashes are not promised across machines; numerical/runtime differences may affect reproduction.

## How the files connect

- `src/acquire_data.py`: download official data, verify archive checksum and save provenance.
- `src/extract_data.py`: parse the observed MySQL dump literal format into issue/project/duplicate-link CSVs; not a general SQL parser.
- `src/audit_data.py`: report counts, missing labels and source flags without cleaning the data.
- `src/prepare_data.py` and `src/data_rules.py`: clean title/description, categorize points, filter and deduplicate, create chronological splits and isolation evidence.
- `src/train.py`: fit TF-IDF on training only; compare two C values and two alpha values on validation; save family winners and freeze the selected experiment. No train-plus-validation refit.
- `src/evaluate.py` and `src/metrics.py`: evaluate frozen family winners and baseline, save predictions/per-category/per-project metrics, confusion matrices and 12 seeded sampled errors.
- `src/predict.py` and `app.py`: load the saved selected pipeline and serve the local form.
- `tests/`: data-rule/parser/isolation checks and saved-model/Streamlit integration checks.

## Data decisions and limitations

Only finite positive labels through 100 are in scope; zero semantics are unverified and zeros are excluded. Fractional positives are retained. Require all three source text/estimate-change flags to equal zero, remove conflicting normalized-text categories, and retain representatives of exact-text and transitive known-duplicate groups before splitting. Split isolation checks ID, issue key, text hash and duplicate group; no semantic near-duplicate detector is implemented.

All description-change flags are zero in the source, so they do not prove estimation-time descriptions. Estimation timestamps equal creation or creation minus one hour for in-scope positives; creation date is used instead. Text cleaning removes detected numeric point-disclosure lines, not all possible leakage. Attachments and linked content are not inputs. Default tokenization loses some technical punctuation and single-character terms.

The chronological test has a changing project/class mixture and is dominated by DM and EVG. This does not establish generalization to unseen teams, exact-point accuracy, or improved sprint planning. TF-IDF plus Naive Bayes learns lexical category associations rather than reasoning about work scope. The app is for local research and discussion, without public deployment.

## Experiment evidence

- `reports/data_manifest.json`, `data_audit.json`, `split_metadata.json`: provenance, data counts and preparation decisions.
- `reports/validation_metrics.json`, `frozen_experiment.json`, `test_metrics.json`: selection, fingerprints and final measurements.
- Generated `reports/test_predictions.csv`, `error_samples.csv` and `confusion_matrix.png`: predictions, sampled mistakes and category-confusion plots (ignored by Git).

## Source attribution

Tawosi, Vali; Alsubaihin, Afnan Abdulaziz A; Moussa, Rebecca; Sarro, Federica (2022). *The TAWOS dataset*. University College London. Dataset. [doi:10.5522/04/21308124.v1](https://doi.org/10.5522/04/21308124.v1).

Source documentation: [SOLAR-group/TAWOS](https://github.com/SOLAR-group/TAWOS). Dataset documentation version 1.1 and repository article version 1 are distinct identifiers. Source license: Apache 2.0; source README additionally specifies research purposes, no harmful use and no reidentification. Source terms/license are preserved under ignored `data/raw/`. This README records those source statements; it does not grant additional rights or a license to this repository's new code.
