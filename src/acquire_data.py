"""Download the official archive and preserve provenance; never execute its SQL."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    url = "https://api.figshare.com/v2/articles/21308124/versions/1"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    (raw / "figshare_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name, url_ in {
        "TAWOS_README.md": "https://raw.githubusercontent.com/SOLAR-group/TAWOS/main/README.md",
        "TAWOS_Licence.txt": "https://raw.githubusercontent.com/SOLAR-group/TAWOS/main/Licence.txt",
        "schema.sql": "https://raw.githubusercontent.com/SOLAR-group/TAWOS/main/TAWOS_Database_Schema_Creation_Script.sql",
    }.items():
        with urllib.request.urlopen(url_, timeout=60) as response:
            (raw / name).write_bytes(response.read())
    source = next(f for f in metadata["files"] if f["name"] == "TAWOS.sql.zip")
    path = raw / source["name"]
    if not path.exists() or path.stat().st_size != source["size"]:
        temporary = path.with_suffix(".zip.part")
        with urllib.request.urlopen(source["download_url"], timeout=120) as response, temporary.open("wb") as out:
            count = 0
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
                count += len(chunk)
                if count % (64 * 1024 * 1024) < len(chunk):
                    print(f"Downloaded {count:,} / {source['size']:,} bytes", flush=True)
        temporary.replace(path)
    digests = {name: hashlib.new(name) for name in ("md5", "sha256")}
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            for digest in digests.values():
                digest.update(chunk)
    assert path.stat().st_size == source["size"], "Download size mismatch"
    assert digests["md5"].hexdigest() == source["computed_md5"], "Download checksum mismatch"
    with zipfile.ZipFile(path) as archive:
        members = [{"name": f.filename, "bytes": f.file_size} for f in archive.infolist()]
    manifest = {
        "source": metadata["url_public_html"], "doi": metadata["doi"],
        "metadata_url": url, "article_version": metadata["version"], "dataset_version": "1.1",
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "archive": source, "checksums": {k: v.hexdigest() for k, v in digests.items()},
        "archive_members": members, "license": metadata["license"],
        "terms_url": "https://github.com/SOLAR-group/TAWOS#terms-of-use",
        "terms_note": "Research purposes only; no harmful use or reidentification. Apache 2.0.",
        "citation": metadata["citation"],
    }
    (reports / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
