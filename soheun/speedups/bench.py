"""Time one worker of the #4 independent training path, and fingerprint it.

The trainer setup is transcribed from artifacts/train_stage.py (resident
Step-3 path): raw banks on the GPU behind _IndexedSplit views, identity-derived
initialization, IndependentStackedDataModule, the same callbacks, checkpoint IO
and Trainer arguments, under the same runtime_policy('cuda').

    python speedups/bench.py --data MEMBERS --members 0,1,2,3,4 --out RUN \
        --patches nosync,fast_gbn,fast_reinforce,graphs

It writes timing.json (per-epoch wall, train, validation and checkpoint time)
and fingerprint.json (per-member digests of final and best weights, Adam and
scheduler state, validation/LR history and probe logits) for compare.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

SOHEUN = pathlib.Path(__file__).resolve().parents[1]
for p in (str(SOHEUN), str(SOHEUN / 'speedups')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(SOHEUN)

import torch  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402

import patches  # noqa: E402


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(sha(v).encode() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


def optimizer_digest(opt) -> str:
    h = hashlib.sha256()
    sd = opt.state_dict()
    for pid in sorted(sd['state']):
        for k in sorted(sd['state'][pid]):
            v = sd['state'][pid][k]
            h.update(f'{pid}.{k}'.encode())
            h.update(sha(v).encode() if torch.is_tensor(v) else repr(v).encode())
    h.update(repr([{k: v for k, v in g.items() if k != 'params'} for g in sd['param_groups']]).encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', required=True, help='comma-separated member indices')
    ap.add_argument('--data', required=True, help='directory of member_NNN.pt files from extract_members.py')
    ap.add_argument('--out', required=True, help='results directory (fingerprint + timings)')
    ap.add_argument('--ckpt-root', default=None, help='where checkpoints go (default: <out>/ckpt)')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patches', default='')
    ap.add_argument('--probe', type=int, default=20000)
    ap.add_argument('--stop-after', type=int, default=None,
                    help='stop after this many completed epochs (then rerun with --resume)')
    ap.add_argument('--resume', action='store_true', help='resume from <ckpt-root>/last.ckpt')
    ap.add_argument('--step2', action='store_true',
                    help='Step-2 AttentionClassifier on prepared transformed members '
                         '(member-<i>.pt with train/val/hparams), trained as train_stage does for '
                         'transformed data: GPU-resident TensorDatasets, no raw banks')
    ap.add_argument('--fixed-batch', type=int, default=None,
                    help='timing only: one batch size for every epoch (no milestones)')
    args = ap.parse_args()

    # stage_processes._initialize
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.autograd.set_detect_anomaly(False)

    names = [p for p in args.patches.split(',') if p]
    patches.install(names)

    from independent_data import IndependentStackedDataModule, stream_identity
    from member_initialization import initialize_members
    from stacked_fvt import StackedFvTClassifier
    from stacked_attention_classifier import StackedAttentionClassifier
    from torch.utils.data import TensorDataset
    import phase3.resumable as resumable
    from phase3.resumable import AtomicCheckpointIO, ResumableIndividualSaver, RngStateCallback, StopAfterEpoch
    from artifacts.train_stage import _IndexedSplit, _TrainingHistory
    from artifacts.runtime_policy import runtime_policy

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    root = pathlib.Path(args.ckpt_root) if args.ckpt_root else out / 'ckpt'
    root.mkdir(parents=True, exist_ok=True)

    # --- checkpoint-write timing (both writers go through atomic_torch_save)
    save_s = {'total': 0.0, 'calls': 0, 'per_epoch': {}}
    original_save = resumable.atomic_torch_save

    def timed_save(obj, path):
        t0 = time.perf_counter()
        original_save(obj, path)
        dt = time.perf_counter() - t0
        save_s['total'] += dt
        save_s['calls'] += 1
        e = len(clock.rows) - 1
        save_s['per_epoch'][e] = save_s['per_epoch'].get(e, 0.0) + dt
    resumable.atomic_torch_save = timed_save

    idx = [int(i) for i in args.members.split(',')]
    pattern = 'member-{}.pt' if args.step2 else 'member_{:03d}.pt'
    records = [torch.load(pathlib.Path(args.data) / pattern.format(i), weights_only=False) for i in idx]
    hps = [r['hparams'] for r in records]
    hp = hps[0]
    if not args.step2 and any(h.get('step') != 3 for h in hps):
        raise ValueError('Old proxy records lack Step-3 identity; extract into a new directory')
    if args.step2 and hp.get('model') != 'AttentionClassifier':
        raise ValueError('--step2 expects AttentionClassifier members')

    class EpochClock(pl.Callback):
        def __init__(self):
            self.rows = []

        def on_train_epoch_start(self, trainer, module):
            now = time.perf_counter()
            if self.rows and 'end' not in self.rows[-1]:
                self.rows[-1]['end'] = now
            self.rows.append({'epoch': int(trainer.current_epoch), 'start': now,
                              'batch_size': int(module.datamodule.batch_size)})

        def on_validation_start(self, trainer, module):
            self.rows[-1]['val_start'] = time.perf_counter()

        def on_validation_end(self, trainer, module):
            self.rows[-1]['val_end'] = time.perf_counter()

        def finish(self):
            now = time.perf_counter()
            if self.rows and 'end' not in self.rows[-1]:
                self.rows[-1]['end'] = now

    clock = EpochClock()

    with runtime_policy('cuda'):
        banks, pairs = [], []
        if args.step2:
            # train_stage, transformed path: each member keeps its own encoded and
            # smeared tensors; the data module places them on the GPU.
            for r in records:
                pairs.append((TensorDataset(*r['train']), TensorDataset(*r['val'])))
        else:
            for r in records:
                (xt, yt, wt), (xv, yv, wv) = r['train'], r['val']
                bank = tuple(torch.cat((a, b), 0).to('cuda') for a, b in ((xt, xv), (yt, yv), (wt, wv)))
                nt, nv = len(yt), len(yv)
                banks.append(bank)
                pairs.append((_IndexedSplit(bank, torch.arange(nt)), _IndexedSplit(bank, torch.arange(nt, nt + nv))))
        run_names = [f'member_{i:03d}' for i in idx]
        kwargs = dict(num_stacks=len(records), num_classes=2, dim_quadjet_features=hp['dim_quadjet_features'],
                      run_names=run_names, stacked_run_name='artifact-stage', device='cpu', depth=hp['depth'])
        if args.step2:
            model = StackedAttentionClassifier(**kwargs)
            members = model.attention_classifiers
        else:
            model = StackedFvTClassifier(**kwargs, dim_input_jet_features=4,
                                         dim_dijet_features=hp['dim_dijet_features'], repr_norm=hp.get('repr_norm', False))
            members = model.fvt_classifiers
        initialize_members(members, hps)
        model.optimizer_config = hp['optimizer']
        model.lr_scheduler_config = hp['lr_scheduler']
        model.execution_chunk_size = 0
        dc = dict(hp['dataloader'])
        if args.fixed_batch:
            dc['batch_size'], dc['batch_size_milestones'] = args.fixed_batch, []
        dm = IndependentStackedDataModule([p[0] for p in pairs], [p[1] for p in pairs], dc['batch_size'],
                                          shuffle_seeds=[h['train_seed'] for h in hps],
                                          estimator_ids=[stream_identity(h) for h in hps],
                                          batch_size_milestones=dc.get('batch_size_milestones', []),
                                          batch_size_multiplier=dc.get('batch_size_multiplier', 2),
                                          num_workers=0, storage_device='cuda' if args.step2 else None)
        model.datamodule = dm
        saver = ResumableIndividualSaver(save_dir=root / 'models', run_names=run_names,
                                         monitor_metrics=[f'val_loss_stack_{i}' for i in range(len(run_names))],
                                         model=hp['model'])
        history = _TrainingHistory()
        callbacks = [clock, saver, RngStateCallback(), history,
                     ModelCheckpoint(dirpath=root, save_last=True, save_top_k=0, save_on_train_epoch_end=True)]
        if args.stop_after is not None:
            callbacks.append(StopAfterEpoch(args.stop_after - 1))
        trainer = pl.Trainer(accelerator='gpu', devices=1, precision='32-true', max_epochs=args.epochs,
                             logger=False, callbacks=callbacks, plugins=[AtomicCheckpointIO()],
                             num_sanity_val_steps=0, reload_dataloaders_every_n_epochs=1,
                             enable_progress_bar=False, enable_model_summary=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        trainer.fit(model, datamodule=dm, ckpt_path=str(root / 'last.ckpt') if args.resume else None)
        torch.cuda.synchronize()
        fit_s = time.perf_counter() - t0
        clock.finish()
        if args.stop_after is not None:
            print(f"[bench] stopped after {args.stop_after} epochs; rerun with --resume", flush=True)
            return

        epochs = []
        for r in clock.rows:
            e = {'epoch': r['epoch'], 'batch_size': r['batch_size'], 'wall_s': r['end'] - r['start']}
            if 'val_start' in r:
                e['train_s'] = r['val_start'] - r['start']
                e['val_s'] = r['val_end'] - r['val_start']
                e['after_val_s'] = r['end'] - r['val_end']
            e['ckpt_s'] = save_s['per_epoch'].get(len(epochs), 0.0)
            epochs.append(e)
        import independent_training
        timing = {'members': idx, 'patches': names, 'fixed_batch': args.fixed_batch, 'ckpt_root': str(root),
                  'epochs_run': args.epochs, 'fit_s': fit_s, 'checkpoint_write_s': save_s['total'],
                  'checkpoint_writes': save_s['calls'], 'epochs': epochs,
                  'peak_alloc_gb': torch.cuda.max_memory_allocated() / 1e9,
                  'graph_stats': getattr(independent_training, 'GRAPH_STATS', None),
                  'node': os.uname().nodename, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                  'mps': bool(os.environ.get('CUDA_MPS_PIPE_DIRECTORY'))}
        (out / 'timing.json').write_text(json.dumps(timing, indent=1, sort_keys=True))
        print(f"[bench] members={idx} patches={names} fit {fit_s:.1f}s ckpt {save_s['total']:.1f}s/{save_s['calls']}", flush=True)

        # ---------------------------------------------------------- fingerprint
        # Lightning's teardown moves the module back to the CPU after fit.
        model.to('cuda')
        model.eval()
        fp = {'members': []}
        for i, (name, member) in enumerate(zip(run_names, members)):
            best = torch.load(root / 'models' / f'{name}_best.pt', map_location='cpu', weights_only=False)
            vd = dm.val_datasets[i]  # as the data module placed it (GPU-resident)
            n_probe = min(args.probe, len(vd))
            probe = (vd.gather(torch.arange(n_probe, device='cuda'))[0] if hasattr(vd, 'gather')
                     else vd.tensors[0][:n_probe].to('cuda'))
            with torch.no_grad():
                pred = member(probe)
            fp['members'].append({
                'name': name,
                'final_state': state_digest(member.state_dict()),
                'best_state': state_digest(best),
                'optimizer': optimizer_digest(trainer.optimizers[i]),
                'scheduler': repr(sorted(trainer.lr_scheduler_configs[i].scheduler.state_dict().items())),
                'history': [(r['epoch'], r['batch_size'], r['members'][i]['val_loss'].hex(),
                             r['members'][i]['lr'].hex()) for r in history.records],
                'best_epoch': saver.best_epochs[f'val_loss_stack_{i}'],
                'probe_logits': sha(pred),
            })
        (out / 'fingerprint.json').write_text(json.dumps(fp, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
