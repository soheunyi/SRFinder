"""Verification receipts: skip repeated full verification only while nothing changed.

On a completed CPU fixture campaign: a cold reader fully verifies every case and writes
receipts; a warm reader performs no full verification and returns identical values;
touching or corrupting a file forces full verification (and corruption still fails);
use_receipts=False and a changed verifier-code hash always force full verification.
Publication writes the receipt, so the first read after training is already warm, and
receipts of two verifier versions coexist instead of overwriting each other.
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'phase5')]
from artifacts.training_store import TrainingStore, canonical
from artifacts.campaign_runtime import run_campaign, import_sources, prepare_execution, selected_nodes
import artifacts.case_registry as cr
from test_campaign_runtime import fixture


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    plan = fixture(args.out); store = TrainingStore(args.out / 'store'); import_sources(store, plan)
    root = args.out / 'execution'
    prepare_execution(store, plan, root, device='cpu', export_batch_size=128)
    done = run_campaign(store, plan, root, nproc=1, resume=True, device='cpu', export_batch_size=128)
    assert done['status'] == 'CAMPAIGN_ARTIFACTS_COMPLETE'
    ids = [n['id'] for n in selected_nodes(plan)]

    calls = []
    original = cr.CaseRegistry._verify
    def counting(self, case_id, *a, **k):
        calls.append(case_id); return original(self, case_id, *a, **k)
    cr.CaseRegistry._verify = counting
    def read_all(use_receipts=True):
        calls.clear(); reg = cr.CaseRegistry(store, plan, root / 'registry', resume=True); reg.use_receipts = use_receipts
        values = {k: reg.get(k) for k in ids}; return values, len(calls)

    # publication wrote the receipts: the first read after training verifies nothing
    published, n_published = read_all()
    assert n_published == 0, n_published

    receipts = root / 'verification-receipts'
    shutil.rmtree(receipts)
    cold, n_cold = read_all()
    assert n_cold == len(ids) and len(list(receipts.glob('*.json'))) == len(ids), (n_cold, len(ids))
    size = max(p.stat().st_size for p in receipts.glob('*.json'))
    warm, n_warm = read_all()
    assert n_warm == 0 and canonical(warm) == canonical(cold), n_warm
    assert canonical(published) == canonical(cold)

    # touching one Step-1 output invalidates that case and everything depending on it
    step1 = next(n for n in selected_nodes(plan) if n['stage'] == 1)
    blob = store.blobs / store.read(cold[step1['id']]['completion']['model_ids'][0], 'model')['payload']['filename']
    st = blob.stat(); os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    touched, n_touched = read_all()
    assert 0 < n_touched < len(ids) and step1['id'] in set(calls) and canonical(touched) == canonical(cold), (n_touched, len(ids))
    again, n_again = read_all()
    assert n_again == 0, n_again

    # full mode and a changed verifier both force full verification
    _, n_full = read_all(use_receipts=False); assert n_full == len(ids), n_full
    cr._VERIFIER_SHA = 'changed'; _, n_code = read_all(); assert n_code == len(ids), n_code
    cr._VERIFIER_SHA = None; read_all()
    # both versions now hold receipts; switching between them verifies nothing
    cr._VERIFIER_SHA = 'changed'; _, n_other = read_all(); cr._VERIFIER_SHA = None; _, n_back = read_all()
    assert n_other == 0 and n_back == 0, (n_other, n_back)

    # corrupting content (same size) is caught by the full check the stale receipt forces
    data = bytearray(blob.read_bytes()); data[-1] ^= 0xFF; blob.write_bytes(bytes(data))
    try:
        read_all()
    except ValueError as exc:
        caught = str(exc)
    else:
        raise AssertionError('Corrupted artifact passed verification')
    print(json.dumps({'status': 'PASS', 'cases': len(ids), 'cold_full_verifications': n_cold, 'warm_full_verifications': n_warm,
                      'after_touch_full_verifications': n_touched, 'receipt_bytes_max': size,
                      'use_receipts_false_full_verifications': n_full, 'changed_verifier_full_verifications': n_code,
                      'published_full_verifications': n_published, 'version_switch_full_verifications': n_other + n_back,
                      'corruption_rejected': caught[:80]}), flush=True)


if __name__ == '__main__':
    main()
