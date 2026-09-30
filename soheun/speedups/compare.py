"""Compare fingerprints of bench.py runs against the first one."""
import json
import pathlib
import sys

runs = [pathlib.Path(p) for p in sys.argv[1:]]
ref = json.loads((runs[0] / 'fingerprint.json').read_text())
ok = True
for run in runs[1:]:
    try:
        fp = json.loads((run / 'fingerprint.json').read_text())
    except FileNotFoundError:
        print(f'{run.name}: NO FINGERPRINT (failed run)'); ok = False; continue
    diffs = []
    for a, b in zip(ref['members'], fp['members']):
        for k in a:
            if a[k] != b[k]:
                diffs.append(f"{a['name']}.{k}")
    print(f"{run.name}: {'IDENTICAL' if not diffs else 'DIFFERS ' + ', '.join(diffs)}")
    ok &= not diffs
for run in runs:
    t = run / 'timing.json'
    if t.exists():
        d = json.loads(t.read_text())
        print(f"  {run.name:16s} fit {d['fit_s']:7.1f}s  ckpt {d['checkpoint_write_s']:6.1f}s in {d['checkpoint_writes']} writes")
sys.exit(0 if ok else 1)
