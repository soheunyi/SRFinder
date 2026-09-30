"""Process-local raw-file snapshot cache for bulk source verification."""
from pathlib import Path
from .source_context import _file_sha


def _stamp(path):
    s=path.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


class SourceFingerprints:
    def __init__(self):self.files={}

    def sha256(self,path):
        path=Path(path).resolve();before=_stamp(path)
        if path in self.files:
            stamp,digest=self.files[path]
            if before!=stamp:raise ValueError('Raw source changed during verification')
            return digest
        digest=_file_sha(path)
        if _stamp(path)!=before:raise ValueError('Raw source changed while hashing')
        self.files[path]=(before,digest)
        return digest

    def verify_unchanged(self):
        for path,(stamp,_) in self.files.items():
            if _stamp(path)!=stamp:raise ValueError('Raw source changed during verification')

    def manifest(self):
        self.verify_unchanged()
        return [{'path':str(path),'sha256':digest,'bytes':stamp[2]}
                for path,(stamp,digest) in sorted(self.files.items())]
