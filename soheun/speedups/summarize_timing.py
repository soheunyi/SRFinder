"""Summarize timing-<job>/<config>/bs<B>/w<k>/timing.json into one table."""
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for cfg in sorted(p for p in root.iterdir() if p.is_dir()):
    for bsdir in sorted(cfg.glob('bs*')):
        ws = [json.loads(t.read_text()) for t in sorted(bsdir.glob('w*/timing.json'))]
        if not ws:
            print(f'{cfg.name} {bsdir.name}: no results'); continue
        # steady epochs: drop epoch 0 when more than one ran
        def steady(w):
            e = w['epochs'][1:] if len(w['epochs']) > 1 else w['epochs']
            return e
        ep_wall = statistics.mean(statistics.mean(e['wall_s'] for e in steady(w)) for w in ws)
        train = statistics.mean(statistics.mean(e.get('train_s', 0) for e in steady(w)) for w in ws)
        val = statistics.mean(statistics.mean(e.get('val_s', 0) for e in steady(w)) for w in ws)
        after = statistics.mean(statistics.mean(e.get('after_val_s', 0) for e in steady(w)) for w in ws)
        ckpt = statistics.mean(statistics.mean(e['ckpt_s'] for e in steady(w)) for w in ws)
        P = len(ws)
        members = sum(len(w['members']) for w in ws)
        # throughput: member-epochs per second across the GPU
        thr = members / ep_wall
        rows.append((cfg.name, bsdir.name, P, ep_wall, train, val, after, ckpt, thr))
base = {r[1]: r[8] for r in rows if r[0] == 'P1_base'}
print(f"{'config':20s} {'batch':8s} {'P':>2s} {'epoch_s':>8s} {'train':>7s} {'val':>6s} {'after':>6s} {'ckpt':>6s} {'mem-ep/s':>9s} {'vs P1':>6s}")
for r in rows:
    rel = r[8] / base[r[1]] if r[1] in base else float('nan')
    print(f"{r[0]:20s} {r[1]:8s} {r[2]:2d} {r[3]:8.2f} {r[4]:7.2f} {r[5]:6.2f} {r[6]:6.2f} {r[7]:6.2f} {r[8]:9.3f} {rel:6.2f}x")
