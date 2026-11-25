#!/usr/bin/env python3
"""Verify the complete archive inventory using only the Python standard library."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'archive_manifest.json').read_text())
    expected = {entry['path'] for entry in manifest['files']}
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts
              and p != root / 'archive_manifest.json'}
    if actual != expected:
        raise ValueError({'missing': sorted(expected - actual),
                          'unexpected': sorted(actual - expected)})
    for entry in manifest['files']:
        path = root / entry['path']
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Archive mismatch: ' + entry['path'])
    print(json.dumps({'passed': True, 'hash_verified_files': len(expected)}))


if __name__ == '__main__':
    main()
