"""Compare complete per-member fingerprints, rejecting missing/duplicate members."""
import json
import pathlib
import sys

def differences(reference, candidate):
    def indexed(value):
        rows=value['members']
        by_name={r['name']:r for r in rows}
        if not rows or len(by_name)!=len(rows):raise ValueError('Empty or duplicate member list')
        return by_name
    a,b=indexed(reference),indexed(candidate)
    if a.keys()!=b.keys():return ['member_set']
    return [f'{name}.{key}' for name in sorted(a)
            for key in sorted(a[name].keys()|b[name].keys())
            if key not in a[name] or key not in b[name] or a[name][key]!=b[name][key]]

def main():
    runs=[pathlib.Path(p) for p in sys.argv[1:]]
    if len(runs)<2:raise SystemExit('Provide a reference and at least one candidate')
    ref=json.loads((runs[0]/'fingerprint.json').read_text())
    ok=True
    for run in runs[1:]:
        try:diffs=differences(ref,json.loads((run/'fingerprint.json').read_text()))
        except (FileNotFoundError,ValueError,KeyError) as exc:
            print(f'{run.name}: INVALID FINGERPRINT: {exc}');ok=False;continue
        print(f"{run.name}: {'IDENTICAL' if not diffs else 'DIFFERS '+', '.join(diffs)}")
        ok &= not diffs
    sys.exit(0 if ok else 1)

if __name__=='__main__':main()
