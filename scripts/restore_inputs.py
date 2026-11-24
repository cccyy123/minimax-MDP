"""Restore the five large experiment inputs from the compact Git archive."""

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "inputs" / "manifest.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(root=ROOT, verify_existing=False):
    entries = json.loads((root / "data" / "inputs" / "manifest.json").read_text())
    for entry in entries:
        source = root / entry["compressed"]
        target = root / entry["path"]
        if target.exists():
            if target.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"Existing input has unexpected size: {target}")
            if verify_existing and digest(target) != entry["sha256"]:
                raise RuntimeError(f"Existing input has unexpected checksum: {target}")
            print("Present:", target.relative_to(root), flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", delete=False) as out:
                temporary = Path(out.name)
                hasher = hashlib.sha256()
                total = 0
                with lzma.open(source, "rb") as compressed:
                    while block := compressed.read(1024 * 1024):
                        out.write(block)
                        hasher.update(block)
                        total += len(block)
            if total != entry["bytes"] or hasher.hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Compressed input failed integrity check: {source}")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print("Restored:", target.relative_to(root), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true", help="Hash existing input files as well.")
    args = parser.parse_args()
    restore(verify_existing=args.verify_existing)
