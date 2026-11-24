"""Build a compact review ZIP with restorable compressed inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.git', '.venv', '__pycache__', '.DS_Store', '.matplotlib'}
LARGE_RAW_INPUTS = {entry['path'] for entry in json.loads((ROOT/'data/inputs/manifest.json').read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Write the ZIP outside the repository.')
    if output.exists():
        parser.error('Output exists; choose a new filename.')
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and p.relative_to(ROOT).as_posix() not in LARGE_RAW_INPUTS
                   and not any(part in EXCLUDE for part in p.relative_to(ROOT).parts)
                   and p.suffix not in {'.pyc', '.zip'})
    manifest = []
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            with path.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            manifest.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': digest})
            archive.write(path, 'minimax-MDP-replication/' + rel)
        archive.writestr('minimax-MDP-replication/MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError('ZIP integrity check failed: ' + bad)
    with output.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(digest + '  ' + output.name + '\n')
    print(f'Created and CRC-checked {output}: {len(files)} files, {output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
