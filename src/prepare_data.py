"""Audit exclusions, deduplicate, and save isolated chronological splits."""
import hashlib
import pandas as pd
from src.audit_data import load_issues, counts
from src.common import ROOT, config, sha256, write_json
from src.data_rules import combine_text, text_key, label_category, UnionFind, assert_split_isolation, DISCLOSURE

def main():
    if (ROOT / "reports/test_metrics.json").exists():
        raise SystemExit("Final test already evaluated. Archive this experiment before preparing a new one.")
    cfg = config()
    data = load_issues()
    initial = len(data)
    exclusions = {}
    removed = []

    def exclude(mask, reason):
        nonlocal data
        bad = data.loc[mask, ["ID", "Issue_Key"]].copy()
        bad["reason"] = reason
        removed.append(bad)
        exclusions[reason] = len(bad)
        data = data.loc[~mask].copy()

    exclude(data.ID.duplicated() | data.Issue_Key.duplicated(), "duplicate_identity")
    data["points"] = pd.to_numeric(data.Story_Point, errors="coerce")
    exclude(data.Story_Point.eq(""), "missing_label")
    exclude(data.points.isna(), "nonnumeric_label")
    exclude(data.points.le(0), "nonpositive_label")
    exclude(data.points.gt(cfg["valid_story_points"]["maximum_inclusive"]), "outside_declared_point_scope")
    positive_in_scope = len(data)
    exclude(~data.filter(like="Changed_After_Estimation").eq("0").all(axis=1), "changed_or_unknown_text_or_estimate")
    unchanged = len(data)
    data["date"] = pd.to_datetime(data[cfg["split_timestamp"]], errors="coerce", utc=True)
    exclude(data.date.isna(), "missing_or_invalid_creation_date")
    data["label"] = data.points.map(lambda p: label_category(p, cfg["category_upper_bounds"]))
    disclosure_count = int((data.Title.map(lambda s: bool(DISCLOSURE.search(s))) | data.Description.map(lambda s: bool(DISCLOSURE.search(s)))).sum())
    data["text"] = [combine_text(a, b) for a, b in zip(data.Title, data.Description)]
    data["text_hash"] = data.text.map(lambda s: hashlib.sha256(text_key(s).encode()).hexdigest())
    exclude(data.text.map(text_key).str.len().eq(0), "empty_cleaned_text")
    text_conflicts = data.groupby("text_hash").label.nunique()
    conflict_hashes = set(text_conflicts[text_conflicts.gt(1)].index)
    exclude(data.text_hash.isin(conflict_hashes), "identical_text_conflicting_categories")
    data = data.sort_values(["date", "ID"], kind="stable")
    exclude(data.text_hash.duplicated(), "duplicate_normalized_text")

    groups = UnionFind()
    links = pd.read_csv(ROOT / "data/extracted/Issue_Link.csv", dtype=str, keep_default_na=False)
    for row in links.itertuples(index=False):
        if row.Issue_ID and row.Target_Issue_ID:
            groups.union(row.Issue_ID, row.Target_Issue_ID)
    data["duplicate_group"] = data.ID.map(groups.find)
    # Retain the earliest eligible representative per known duplicate group.
    exclude(data.duplicate_group.duplicated(), "known_duplicate_link_group")
    data = data.sort_values(["date", "ID"], kind="stable").reset_index(drop=True)
    n = len(data)
    train_target = int(n * cfg["split"]["train_fraction"])
    val_target = int(n * (cfg["split"]["train_fraction"] + cfg["split"]["validation_fraction"]))
    train_end = data.date.iloc[train_target]
    val_end = data.date.iloc[val_target]
    # All records with the same timestamp stay together; future dates cannot
    # enter training due to a tie broken by arbitrary issue ID order.
    data["split"] = "test"
    data.loc[data.date.lt(val_end), "split"] = "validation"
    data.loc[data.date.lt(train_end), "split"] = "train"
    assert_split_isolation(data)
    assert len(data[data.split.eq("test")]) >= cfg["split"]["minimum_test"], "Revise scope before modeling: insufficient test tickets"
    for name, subset in data.groupby("split"):
        assert set(subset.label) == set(cfg["labels"]), f"Missing class in {name}"
    projects = pd.read_csv(ROOT / "data/extracted/Project.csv", dtype=str, keep_default_na=False)
    data["project"] = data.Project_ID.map(projects.set_index("ID").Project_Key)
    assert data.project.notna().all()
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    columns = ["ID", "Issue_Key", "Project_ID", "project", "points", "label", "date", "text", "text_hash", "duplicate_group", "split"]
    data[columns].to_csv(out / "tickets.csv", index=False)
    for name, subset in data.groupby("split"):
        subset[columns].to_csv(out / f"{name}.csv", index=False)
    pd.concat(removed).to_csv(out / "exclusions.csv", index=False)
    ids = data[[c for c in columns if c != "text"]]
    ids.to_csv(ROOT / "reports/split_ids.csv", index=False)
    metadata = {
        "processed_unique_issues": initial, "positive_labels_in_scope": positive_in_scope,
        "unchanged_before_text_dedup": unchanged, "cleaned_labeled_modeling_records": n,
        "sequential_exclusions": exclusions, "disclosure_affected_records": disclosure_count,
        "conflicting_text_groups_removed": len(conflict_hashes),
        "category_bounds": cfg["category_upper_bounds"], "point_scope": cfg["valid_story_points"],
        "split_timestamp": cfg["split_timestamp"], "method": "global chronological by creation date; ties kept together",
        "train_end_exclusive": train_end.isoformat(), "validation_end_exclusive": val_end.isoformat(),
        "seed": cfg["seed"], "data_sha256": sha256(out / "tickets.csv"), "config_sha256": sha256(ROOT / "config.json"),
        "split_counts": {}, "notes": [
            "Zero meaning is unverified; zero and missing values excluded, never relabeled low.",
            "Positive values above 100 excluded as outside conventional scope; source SQL examples use 1..100; fractions retained.",
            "Estimation dates equal creation or creation minus one hour for every in-scope positive label; creation date chosen instead.",
            "Source flags must all equal zero, but every description-change flag is zero in this dump; this does not prove pre-estimation text.",
            "Current snapshot descriptions may contain post-estimation details undetected by source flags or disclosure rules.",
            "Conflicting normalized text categories removed entirely; earliest representative retained for exact text and known duplicate links.",
            "No near-duplicate semantic matching; residual related-ticket leakage is possible.",
            "Global chronology tests later tickets from an evolving project mixture; no guarantee of unseen-team performance.",
        ],
    }
    for name, subset in data.groupby("split"):
        metadata["split_counts"][name] = {"records": len(subset), "classes": counts(subset.label), "projects": counts(subset.project), "date_min": subset.date.min().isoformat(), "date_max": subset.date.max().isoformat()}
    assert n + sum(exclusions.values()) == initial
    write_json(ROOT / "reports/split_metadata.json", metadata)
    print({k: v for k, v in metadata.items() if k not in ("split_counts", "notes")})
    print({k: {"records": v["records"], "classes": v["classes"]} for k, v in metadata["split_counts"].items()})

if __name__ == "__main__":
    main()
