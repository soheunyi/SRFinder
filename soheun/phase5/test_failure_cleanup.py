"""A failed case must not leave the coordinator waiting on surviving workers.

Lightning installs a SIGTERM handler that keeps a training process alive, so
terminate() alone left pilot job 210877 holding its GPU after a CUDA OOM. Here
one worker fails while the others ignore SIGTERM; the coordinator must re-raise
the failure promptly, record FAILED and leave no live worker processes.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'phase5')]
import artifacts.campaign_runtime as runtime
from artifacts.training_store import TrainingStore
from test_campaign_runtime import fixture


def fake_task(payload):
    marker = Path(os.environ['FAILURE_CLEANUP_MARKER'])
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        signal.signal(signal.SIGTERM, lambda *_: print('Received SIGTERM; ignoring', flush=True))
        time.sleep(600)
        raise AssertionError('Surviving worker was not stopped')
    time.sleep(3)
    raise RuntimeError('Injected worker failure')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    os.environ['FAILURE_CLEANUP_MARKER'] = str(args.out / 'first-task')
    plan = fixture(args.out)
    store = TrainingStore(args.out / 'store')
    runtime.import_sources(store, plan)
    root = args.out / 'execution'
    runtime.prepare_execution(store, plan, root, device='cpu', export_batch_size=128)
    runtime._task = fake_task
    started = time.monotonic()
    try:
        runtime.run_campaign(store, plan, root, nproc=2, resume=True, device='cpu', export_batch_size=128)
    except RuntimeError as exc:
        assert 'Injected worker failure' in str(exc), exc
    else:
        raise AssertionError('Worker failure was not raised')
    elapsed = time.monotonic() - started
    progress = json.loads((root / 'progress.json').read_text())
    assert progress['status'] == 'FAILED' and progress['error_type'] == 'RuntimeError'
    assert not mp.active_children(), 'Worker processes survived the coordinator'
    assert elapsed < 60, f'Coordinator took {elapsed:.0f} s to exit'
    print(json.dumps({'status': 'PASS', 'failure_reraised': True, 'sigterm_ignoring_workers_stopped': True,
                      'coordinator_exit_s': round(elapsed, 1)}), flush=True)


if __name__ == '__main__':
    main()
