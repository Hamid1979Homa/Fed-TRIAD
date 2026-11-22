"""Run the main study (seeds 1-10) and the pre-registered sensitivity sweeps.
One result file per (setting, seed) in <out>/runs/, so the run can be resumed; then one file per setting.
Usage: python run_sweep.py --out ../results [--jobs 2] [--only name1 name2]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import fedtriad_sim as fs

SEEDS = list(range(1, 11))
BASE = dict(m=12, scenario='drift', set={})


def S(**kw):
    d = dict(BASE, set={})
    d.update({k: v for k, v in kw.items() if k in ('m', 'scenario')})
    d['set'] = {k: v for k, v in kw.items() if k not in ('m', 'scenario')}
    return d


SETTINGS = {
    # main study
    'drift_m12': S(), 'drift_m30': S(m=30), 'scarce_m12': S(scenario='scarce'),
    # 1. kappa x omega grid (0.6/0.6 is drift_m12)
    **{f'kw_k{k}_w{w}': S(kappa=k, omega=w) for k in (0.3, 0.6, 1.0) for w in (0.3, 0.6, 1.0)
       if not (k == 0.6 and w == 0.6)},
    # 2. shifted clients (s=8 is drift_m12)
    **{f's{s}': S(s=s) for s in (0, 4, 12)},
    # 3. hysteresis (0.02 is drift_m12)
    **{f'tau{t}': S(tau_imp=t) for t in (0.0, 0.01, 0.05, 0.10)},
    # 4. window (10 is drift_m12)
    **{f'W{w}': S(W=w) for w in (5, 20)},
    # 5. participation (12 and 30 are the main study)
    'm6': S(m=6),
    # 6. data per client (60 is drift_m12)
    **{f'n{n}': S(n_path=n, n_gate=n // 2) for n in (30, 120)},
}


def cfg_of(st):
    c = dict(fs.CFG)
    c.update(fs.scenario_overrides(st['scenario']))
    c.update(st['set'])
    return c


def job(args):
    name, seed, out = args
    path = f'{out}/runs/{name}/seed{seed}.npy'
    if os.path.exists(path):
        return name, seed, 'cached'
    st = SETTINGS[name]
    t0 = time.time()
    r = fs.run(seed, cfg_of(st), st['m'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path + '.tmp.npy', r, allow_pickle=True)
    os.replace(path + '.tmp.npy', path)
    return name, seed, f'{time.time() - t0:.0f}s'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    names = a.only or list(SETTINGS)
    jobs = [(n, s, a.out) for n in names for s in SEEDS]
    print(f'{len(jobs)} runs', flush=True)
    with Pool(a.jobs) as pool:
        for k, (n, s, msg) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f'[{k}/{len(jobs)}] {n} seed {s}: {msg}', flush=True)
    for n in names:
        res = {s: np.load(f'{a.out}/runs/{n}/seed{s}.npy', allow_pickle=True).item() for s in SEEDS}
        np.save(f'{a.out}/{n}.npy', res, allow_pickle=True)
        st = SETTINGS[n]
        json.dump(dict(CFG=cfg_of(st), m=st['m'], seeds=SEEDS, scenario=st['scenario'], overrides=st['set']),
                  open(f'{a.out}/{n}.cfg.json', 'w'), indent=1, default=list)
    print('done', flush=True)


if __name__ == '__main__':
    main()
