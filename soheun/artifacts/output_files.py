"""Publish derived text/CSV files atomically in their owned output directory."""
import csv,io,os,tempfile
from pathlib import Path

def atomic_text(path,text):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,mode='w',delete=False) as handle:
        tmp=Path(handle.name)
        try:
            handle.write(text);handle.flush();os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True);raise
    try:os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)

def atomic_csv(path,rows):
    if not rows:return
    handle=io.StringIO(newline='')
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    atomic_text(path,handle.getvalue())
