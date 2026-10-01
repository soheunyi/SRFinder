"""Process-local invalidation for already fully verified immutable results.

Keeps file stamps, never event arrays or tensors. A hit still checks every
referenced file. Any change invalidates the result by raising; new processes
must perform full content/source verification again.
"""
from pathlib import Path
from .source_fingerprints import _stamp


def _ids(value):
    if isinstance(value, str):
        if len(value) == 64 and all(c in '0123456789abcdef' for c in value):
            yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _ids(key)
            yield from _ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _ids(item)


class VerificationSnapshot:
    def __init__(self, store, result_id, external_paths=()):
        paths = set(Path(p).resolve() for p in external_paths)
        pending = [result_id]
        seen = set()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            path = store.records / (key + '.json')
            # Not every digest is an artifact ID (e.g. raw pool checksums).
            if not path.is_file():
                if key == result_id:
                    raise FileNotFoundError(path)
                continue
            before = _stamp(path)
            record = store.read(key)
            if _stamp(path) != before:
                raise ValueError('Artifact record changed during snapshot')
            paths.add(path.resolve())
            payload = record.get('payload')
            if isinstance(payload, dict) and 'filename' in payload:
                paths.add((store.blobs / payload['filename']).resolve())
            pending.extend(_ids(record))
        self.stamps = {p: _stamp(p) for p in paths}

    def verify_unchanged(self):
        for path, stamp in self.stamps.items():
            try:
                current = _stamp(path)
            except FileNotFoundError as exc:
                raise ValueError('Verified result input disappeared') from exc
            if current != stamp:
                raise ValueError('Verified result input changed')
