"""Deterministic rules shared by preparation and inference."""
import html
import math
import re
import unicodedata

# Remove an entire line with an explicit numeric effort disclosure. This is
# conservative and may remove useful context. It cannot catch every disclosure.
DISCLOSURE = re.compile(r"(?:\bstory[\s_-]*points?\b\s*(?:estimate[ds]?\s*)?[:=\-]?\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:story[\s_-]*points?|SP)\b|\bSP\s*[:=]\s*\d+(?:\.\d+)?)", re.I)

def clean_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = re.sub(r"<[^>]+>", " ", text)
    text = "\n".join(line for line in text.splitlines() if not DISCLOSURE.search(line))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\{(?:code|noformat)(?::[^}]*)?\}", " ", text, flags=re.I)
    text = re.sub(r"\[([^|\]]+)\|[^\]]+\]", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def combine_text(title, description):
    return " ".join(part for part in (clean_text(title), clean_text(description)) if part)

def text_key(text):
    # Stronger than whitespace normalization: punctuation/case-only clones
    # share a key, while technical tokens remain in the model's actual text.
    return " ".join(re.findall(r"\w+", text.casefold()))

def label_category(value, bounds=None):
    bounds = bounds or {"low": 2, "medium": 5}
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    if number <= bounds["low"]:
        return "low"
    return "medium" if number <= bounds["medium"] else "high"

def assert_split_isolation(frame):
    for column in ("ID", "Issue_Key", "text_hash", "duplicate_group"):
        assert frame.groupby(column)["split"].nunique().max() == 1, f"Cross-split leakage: {column}"
    assert frame["ID"].is_unique and frame["text_hash"].is_unique

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        parent = self.parent.setdefault(key, key)
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)
