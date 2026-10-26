"""Record only installed dependencies reachable from this project's requirements."""
import importlib.metadata as metadata
import subprocess
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from src.common import ROOT

def main():
    todo = [Requirement(line).name for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip()]
    seen = {}
    while todo:
        name = canonicalize_name(todo.pop())
        if name in seen:
            continue
        dist = metadata.distribution(name)
        seen[name] = f"{dist.metadata['Name']}=={dist.version}"
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                todo.append(requirement.name)
    (ROOT / "requirements-lock.txt").write_text("\n".join(seen[key] for key in sorted(seen)) + "\n", encoding="utf-8")
    (ROOT / "reports/environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print(f"Locked {len(seen)} project packages for Python {sys.version.split()[0]}")

if __name__ == "__main__":
    main()
