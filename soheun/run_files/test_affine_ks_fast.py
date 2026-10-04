"""Validation gate for the C++ bootstrap of the affine weighted KS test.

1. Deterministic outputs (KS statistic, ks_t, n3/n4, effective sizes) equal the reference.
2. With the reference's NumPy multipliers, the C++ replicate loop reproduces the reference
   exactly: same max_exceedances, p-value and maximizing t, on varied synthetic datasets.
3. With native multipliers: Poisson(1) moments; determinism and thread-count independence;
   p-values agree with the reference within Monte Carlo error; null rejection rate matches.
"""
import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
reference = load('_gate_compiled', ROOT / 'run_files/affine_weighted_ks_compiled.py')
fast = load('_gate_fast', ROOT / 'run_files/affine_weighted_ks_fast.py')
L, U = 0.4, 10.0


def dataset(rng, n3, n4, shift=0.0, tilt=0.0, ties=False):
    """Scores crowded just above L, like the SR; optional 4b shift/tilt for alternatives."""
    z3 = L + rng.exponential(0.08, n3); z4 = L + rng.exponential(0.08 * (1 + shift), n4)
    if ties: z3, z4 = np.round(z3, 3), np.round(z4, 3)
    z3, z4 = np.clip(z3, L, U), np.clip(z4, L, U)
    w3 = rng.gamma(4.0, 0.25, n3); w4 = rng.gamma(4.0, 0.25, n4) * (1 + tilt * (z4 - L))
    return z3, w3, z4, w4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); args = ap.parse_args()
    rng = np.random.default_rng(20261003); report = {}

    # 1 + 2: exact agreement with the reference stream
    cases = [dict(n3=800, n4=1500), dict(n3=3000, n4=5000, shift=0.05), dict(n3=2000, n4=2500, tilt=0.3),
             dict(n3=1500, n4=1500, ties=True), dict(n3=400, n4=9000, shift=-0.03), dict(n3=5000, n4=4000, tilt=-0.2)]
    exact = []
    for i, c in enumerate(cases):
        data = dataset(rng, **c)
        for B, seed in ((200, 7 + i), (99, 1729)):
            a = reference.affine_ks_test(*data, L=L, U=U, B=B, alpha=0.05, seed=seed)
            b = fast.affine_ks_test(*data, L=L, U=U, B=B, alpha=0.05, seed=seed, multipliers='numpy', threads=4)
            same = asdict(a) == asdict(b)  # different module copies of the same dataclass
            exact.append(same)
            if not same:
                raise AssertionError(f'numpy-multiplier mismatch case {c} B={B}: ref {a} fast {b}')
    report['exact_with_reference_stream'] = f'{sum(exact)}/{len(exact)} identical AffineKSResult'

    # 3a: Poisson(1) sampler
    draws = np.concatenate([fast.poisson_sample(99, s, 200000) for s in range(10)])
    mean, var, p0 = draws.mean(), draws.var(), np.mean(draws == 0)
    se = 1 / np.sqrt(draws.size)
    assert abs(mean - 1) < 5 * se and abs(var - 1) < 5 * np.sqrt(2) * se and abs(p0 - np.exp(-1)) < 5 * 0.5 * se, (mean, var, p0)
    report['poisson'] = {'n': int(draws.size), 'mean': float(mean), 'var': float(var), 'P0': float(p0), 'exp(-1)': float(np.exp(-1))}

    # 3b: determinism and thread independence
    data = dataset(rng, 3000, 4000)
    r1 = fast.affine_ks_test(*data, L=L, U=U, B=300, seed=11, threads=1)
    r8 = fast.affine_ks_test(*data, L=L, U=U, B=300, seed=11, threads=8)
    r1b = fast.affine_ks_test(*data, L=L, U=U, B=300, seed=11, threads=3, chunk=17)
    assert r1 == r8 == r1b, (r1, r8, r1b)
    report['deterministic_and_thread_independent'] = True

    # 3c: native vs reference p-values on the same data, within Monte Carlo error
    n_sets = 12 if args.quick else 30; B = 400; diffs = []
    for i in range(n_sets):
        data = dataset(rng, 1500, 2500, shift=rng.choice([0.0, 0.02, 0.05]))
        a = reference.affine_ks_test(*data, L=L, U=U, B=B, seed=100 + i)
        b = fast.affine_ks_test(*data, L=L, U=U, B=B, seed=100 + i, threads=4)
        p = max(min((a.p_value + b.p_value) / 2, 0.99), 0.01)
        diffs.append((b.p_value - a.p_value) / np.sqrt(2 * p * (1 - p) / B))
    diffs = np.array(diffs)
    assert np.all(np.abs(diffs) < 4.5) and abs(diffs.mean()) < 3 / np.sqrt(n_sets), diffs
    report['native_vs_reference_p'] = {'datasets': n_sets, 'B': B, 'max_abs_z': float(np.abs(diffs).max()), 'mean_z': float(diffs.mean())}

    # 3d: null rejection rate, native vs reference, on identical null datasets
    n_null = 150 if args.quick else 400; B = 199; rej = {'reference': 0, 'native': 0}
    for i in range(n_null):
        data = dataset(rng, 600, 900)
        rej['reference'] += reference.affine_ks_test(*data, L=L, U=U, B=B, seed=5000 + i).reject
        rej['native'] += fast.affine_ks_test(*data, L=L, U=U, B=B, seed=5000 + i, threads=4).reject
    se = np.sqrt(0.05 * 0.95 / n_null)
    assert abs(rej['native'] - rej['reference']) / n_null < 4 * np.sqrt(2) * se, rej
    report['null_rejections'] = {'datasets': n_null, 'alpha': 0.05, **{k: v / n_null for k, v in rej.items()}}

    print(json.dumps({'status': 'PASS', **report}), flush=True)


if __name__ == '__main__':
    main()
