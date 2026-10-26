from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def classification_metrics(y, prediction, labels):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "test_size": len(y),
        "per_class": classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
    }
