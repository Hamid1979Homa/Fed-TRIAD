"""Run the main study (seeds 1–10) and the sensitivity settings.

Use a fresh output directory or pass --resume with the original environment and settings.
"""
import argparse
import json
import os
import platform
import sys
import time
from importlib import metadata
from multiprocessing import get_context
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np
from threadpoolctl import threadpool_info

import fedtriad_sim as fs
from run_files import atomic_output, directory_lock, sha256, write_json

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


def study_manifest(names):
    """Identify the data-generation protocol and the numerical environment used by cached jobs."""
    pools = []
    for lib in threadpool_info():
        pools.append(dict(lib, filepath=Path(lib['filepath']).name))
    pools.sort(key=lambda lib: lib['filepath'])
    packages = ('numpy', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'threadpoolctl')
    return {
        'format': 1,
        'code_sha256': {Path(p).name: sha256(p) for p in (__file__, fs.__file__)},
        'platform': platform.platform(),
        'python': platform.python_version(),
        'packages': {name: metadata.version(name) for name in packages},
        'numpy_cpu_features': sorted(name for name, available in
                                     np._core._multiarray_umath.__cpu_features__.items() if available),
        'libraries': pools,
        'numeric_environment': {name: os.environ.get(name) for name in (
            'OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NPY_DISABLE_CPU_FEATURES',
        )},
        'seeds': SEEDS,
        'settings': {name: setting_config(name) for name in names},
    }


def setting_config(name):
    st = SETTINGS[name]
    cfg = cfg_of(st)
    cfg['ditto_lams'] = list(cfg['ditto_lams'])
    return dict(CFG=cfg, m=st['m'], seeds=SEEDS, scenario=st['scenario'], overrides=st['set'])


def save_run(path, data):
    with atomic_output(path) as tmp:
        with tmp.open('wb') as stream:
            np.save(stream, data, allow_pickle=True)


def prepare_result_dir(directory, names, resume):
    marker = directory / 'study.json'
    manifest = study_manifest(names)
    if marker.exists():
        if not resume:
            raise ValueError(f'{directory} already contains a study. Use --resume or choose a fresh directory.')
        if json.loads(marker.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Run identity differs from study.json. Choose a fresh output directory.')
    else:
        if any(p.name != '.writer.lock' for p in directory.iterdir()):
            raise ValueError(f'{directory} is not empty and has no study manifest. Choose a fresh directory.')
        write_json(marker, manifest)


def job(args):
    name, seed, out = args
    path = out / 'runs' / name / f'seed{seed}.npy'
    digest = path.with_suffix('.json')
    if path.is_file() and digest.is_file():
        try:
            if json.loads(digest.read_text(encoding='utf-8')) == {'sha256': sha256(path)}:
                return name, seed, 'verified cache'
        except (OSError, ValueError):
            pass
    st = SETTINGS[name]
    t0 = time.time()
    r = fs.run(seed, cfg_of(st), st['m'])
    save_run(path, r)
    write_json(digest, {'sha256': sha256(path)})
    return name, seed, f'{time.time() - t0:.0f}s'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--only', nargs='+', choices=SETTINGS)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    names = a.only or list(SETTINGS)
    if a.jobs < 1 or len(set(names)) != len(names):
        ap.error('Use a positive worker count and distinct setting names.')
    jobs = [(n, s, a.out) for n in names for s in SEEDS]
    try:
        with directory_lock(a.out):
            prepare_result_dir(a.out, names, a.resume)
            print(f'{len(jobs)} runs', flush=True)
            with get_context('spawn').Pool(a.jobs) as pool:
                for k, (n, s, msg) in enumerate(pool.imap_unordered(job, jobs), 1):
                    print(f'[{k}/{len(jobs)}] {n} seed {s}: {msg}', flush=True)
            for n in names:
                res = {s: np.load(a.out / 'runs' / n / f'seed{s}.npy', allow_pickle=True).item()
                       for s in SEEDS}
                save_run(a.out / f'{n}.npy', res)
                write_json(a.out / f'{n}.cfg.json', setting_config(n))
    except (ValueError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
