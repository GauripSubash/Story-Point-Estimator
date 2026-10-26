"""Inspect all extracted issue records before selecting model data."""
import pandas as pd
from src.common import ROOT, write_json

def load_issues():
    return pd.read_csv(ROOT / "data/extracted/Issue.csv", dtype=str, keep_default_na=False)

def counts(series):
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}

def main():
    data = load_issues()
    points = pd.to_numeric(data.Story_Point, errors="coerce")
    report = {
        "issue_rows_inspected": len(data), "unique_issue_ids_inspected": data.ID.nunique(),
        "unique_issue_keys": data.Issue_Key.nunique(), "projects": data.Project_ID.nunique(),
        "duplicate_ids": int(data.ID.duplicated().sum()),
        "duplicate_issue_keys": int(data.Issue_Key.duplicated().sum()),
        "missing_by_field": {c: int(data[c].eq("").sum()) for c in data},
        "labels": {"missing": int(data.Story_Point.eq("").sum()), "nonnumeric_nonmissing": int((points.isna() & data.Story_Point.ne("")).sum()), "negative": int(points.lt(0).sum()), "zero": int(points.eq(0).sum()), "positive": int(points.gt(0).sum()), "histogram": counts(data.Story_Point)},
        "flags": {c: counts(data[c]) for c in data if "Changed_After" in c},
        "project_counts": counts(data.Project_ID),
        "project_positive_labels": counts(data.loc[points.gt(0), "Project_ID"]),
        "sample_metadata": data[["ID", "Issue_Key", "Project_ID", "Story_Point", "Creation_Date", "Estimation_Date"]].head(5).to_dict("records"),
    }
    write_json(ROOT / "reports/data_audit.json", report)
    print(report["labels"])
    print({k: report[k] for k in ("issue_rows_inspected", "unique_issue_ids_inspected", "projects", "flags")})

if __name__ == "__main__":
    main()
