"""Parse only required tables from the official mysqldump. SQL is never executed."""
import csv
import io
import re
import zipfile
from collections import Counter
from src.common import ROOT, write_json

# Quoted strings are matched before punctuation, so commas/parentheses inside
# descriptions cannot become SQL delimiters. Unknown syntax fails closed.
TOKEN = re.compile(r"'(?:[^'\\]|\\.|'')*'|NULL|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[(),;]", re.DOTALL)
ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}

def decode_value(token):
    if token == "NULL":
        return None
    if token.startswith("'"):
        return re.sub(r"\\(.)", lambda m: ESCAPES.get(m[1], m[1]), token[1:-1]).replace("''", "'")
    return token

def parse_values(values):
    row = None
    position = 0
    for match in TOKEN.finditer(values):
        if values[position:match.start()].strip():
            raise ValueError(f"Unsupported SQL value near offset {position}")
        position = match.end()
        token = match[0]
        if token == "(":
            assert row is None
            row = []
        elif token == ")":
            assert row is not None
            yield row
            row = None
        elif token not in (",", ";"):
            assert row is not None
            row.append(decode_value(token))
    assert row is None and not values[position:].strip(), "Incomplete INSERT"

ISSUE_FIELDS = ["ID", "Issue_Key", "Project_ID", "Title", "Description", "Story_Point", "Creation_Date", "Estimation_Date", "Title_Changed_After_Estimation", "Description_Changed_After_Estimation", "Story_Point_Changed_After_Estimation"]

def main():
    out = ROOT / "data/extracted"
    out.mkdir(parents=True, exist_ok=True)
    wanted = {"Issue": ISSUE_FIELDS, "Project": ["ID", "Project_Key", "Name", "Repository_ID"], "Issue_Link": ["Issue_ID", "Name", "Target_Issue_ID"]}
    counts = Counter()
    schemas = {}
    with zipfile.ZipFile(ROOT / "data/raw/TAWOS.sql.zip") as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".sql") and not n.startswith("__MACOSX")]
        assert len(members) == 1, members
        files = {table: (out / f"{table}.csv").open("w", encoding="utf-8", newline="") for table in wanted}
        try:
            writers = {table: csv.writer(stream) for table, stream in files.items()}
            for table in wanted:
                writers[table].writerow(wanted[table])
            current = None
            with archive.open(members[0]) as source:
                for line in source:
                    if line.startswith(b"CREATE TABLE `"):
                        current = line.split(b"`")[1].decode()
                        if current in wanted:
                            schemas[current] = []
                    elif current in wanted and line.startswith(b"  `"):
                        schemas[current].append(line.split(b"`")[1].decode())
                    elif line.startswith(b") ENGINE"):
                        current = None
                    elif line.startswith(b"INSERT INTO `"):
                        table = line.split(b"`", 2)[1].decode()
                        if table not in wanted:
                            continue
                        assert table in schemas, f"Missing schema for {table}"
                        indices = [schemas[table].index(field) for field in wanted[table]]
                        values = line.split(b" VALUES ", 1)[1].decode("utf-8")
                        for row in parse_values(values):
                            assert len(row) == len(schemas[table]), (table, len(row), len(schemas[table]))
                            # Only known duplicate links are retained; other links are not inputs.
                            selected = [row[i] for i in indices]
                            if table == "Issue_Link" and "duplicat" not in (selected[1] or "").lower():
                                continue
                            writers[table].writerow(selected)
                            counts[table] += 1
                        if table == "Issue" and counts[table] % 25000 < 500:
                            print(f"Extracted {counts[table]:,} issue records", flush=True)
        finally:
            for stream in files.values():
                stream.close()
    write_json(ROOT / "reports/extraction.json", {"rows_written": dict(counts), "actual_schema": schemas, "field_mapping": wanted, "method": "Strict MySQL literal parser; no SQL execution; only duplicate issue links retained"})
    print(dict(counts))

if __name__ == "__main__":
    main()
