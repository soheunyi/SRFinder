"""Snapshot reuse must detect changed files and never trust stale digests."""
import pathlib,sys,tempfile
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.source_fingerprints import SourceFingerprints


def main():
    with tempfile.TemporaryDirectory() as tmp:
        p=pathlib.Path(tmp)/'pool';p.write_bytes(b'old')
        cache=SourceFingerprints();digest=cache.sha256(p)
        with patch('artifacts.source_fingerprints._file_sha',side_effect=AssertionError('Repeated file read')):
            assert cache.sha256(p)==digest
            cache.verify_unchanged()
        p.write_bytes(b'changed')
        for check in (lambda:cache.sha256(p),cache.verify_unchanged):
            try:check()
            except ValueError:pass
            else:raise AssertionError('Changed pool accepted')
    print('PASS: digest reuse and changed-source rejection')


if __name__=='__main__':main()
