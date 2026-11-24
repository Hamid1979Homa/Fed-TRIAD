"""Checks for the extended simulator. Each check is paired with a planted fault that must make it fail.
Usage: python test_sim.py <path to stored 5-seed drift_m12.npy from the first study>
"""
import copy
import sys

import numpy as np

import fedtriad_sim as fs

FAILED = []


def check(name, ok):
    print(f'{"PASS" if ok else "FAIL"}  {name}')
    if not ok:
        FAILED.append(name)


RTOL = 1e-4   # relative tolerance for the reproduction check (see the note in README.md, "Reproducibility")


def last20(r):
    """The metric of the paper: excess risk averaged over the last 20 rounds, for each first-study variant."""
    return {v: float(np.mean(r['excess'][v][-20:])) for v in fs.VARIANTS}


def same_as_stored(r, ref, rtol=RTOL):
    a, b = last20(r), last20(ref)
    return all(abs(a[v] - b[v]) <= rtol * abs(b[v]) for v in fs.VARIANTS)


def bit_identical(r, ref):
    return all(np.array_equal(r['excess'][v], ref['excess'][v]) for v in fs.VARIANTS)


# 1. Compare seed 1 with the historical study. The smoke test uses a tolerance, while
#    reproduce.py checks the entire archive at the documented precision.
ref = np.load(sys.argv[1], allow_pickle=True).item()[1]
cfg = dict(fs.CFG)
r = fs.run(1, cfg, 12)
check(f'seed 1 reproduces the stored first-study excess risk (mean of the last 20 rounds, relative tolerance {RTOL:g}) '
      'of all 12 variants', same_as_stored(r, ref))
print(f'INFO  curves bit-identical to the stored ones in this environment: {bit_identical(r, ref)}')
bad = dict(cfg, kappa=cfg['kappa'] + 1e-3)                       # planted fault
check('control: a 1e-3 change of kappa breaks the reproduction (expected FAIL -> recorded as PASS if it fails)',
      not same_as_stored(fs.run(1, bad, 12), ref))
check('Ditto is reported for every round', np.all(np.isfinite(r['excess']['Ditto'])) and r['excess']['Ditto'].shape == (cfg['T'],))

# 2. Ditto update: lam = 0 equals plain local SGD on the same random stream; large lam keeps V near G.
rng = np.random.default_rng(0)
F = rng.normal(size=(60, 16)); Y = rng.normal(size=(60, 3)); V0 = rng.normal(size=(3, 16)); Gm = rng.normal(size=(3, 16))
a = fs.ditto_sgd(V0, Gm, F, Y, cfg, np.random.default_rng(5), 0.0)
b = fs.local_sgd(V0, F, Y, cfg, np.random.default_rng(5))
check('Ditto with lambda = 0 equals local SGD', np.allclose(a, b, atol=1e-12))


def drift_to_G(fn):
    V = V0.copy()
    for _ in range(200):
        V = fn(V, Gm, F, Y, cfg, np.random.default_rng(1), 15.0)
    return np.linalg.norm(V - Gm) / np.linalg.norm(V0 - Gm)


check('Ditto with lambda = 15 pulls V towards G (relative distance < 0.2)', drift_to_G(fs.ditto_sgd) < 0.2)


def ditto_wrong_sign(V0_, Gm_, F_, Y_, c, rng_, lam):                # planted fault
    V = V0_.copy()
    for _ in range(c['E']):
        idx = rng_.choice(len(F_), size=min(c['batch'], len(F_)), replace=False)
        Gr = (F_[idx] @ V.T - Y_[idx]).T @ F_[idx] / len(idx) + c['ridge'] * V - lam * (V - Gm_)
        V -= c['lr'] * Gr
    return V


with np.errstate(all='ignore'):
    d = drift_to_G(ditto_wrong_sign)
check('control: a sign-flipped proximal term does NOT pull V towards G', not (d < 0.2))

# 3. Shift size s
w8 = fs.World(1, dict(cfg, s=8))
orig = w8.concept_pre.copy(); orig[0:4] = 2; orig[4:8] = 4
check('s = 8 reproduces the original concept assignment', np.array_equal(w8.concept_post, orig))
w0 = fs.World(1, dict(cfg, s=0))
check('s = 0 leaves every concept unchanged', np.array_equal(w0.concept_post, w0.concept_pre))
w12 = fs.World(1, dict(cfg, s=12))
check('s = 12 leaves only concepts {2, 4} among clients 0-11', set(w12.concept_post[:12]) == {2, 4})
check('control: s = 4 is not mistaken for s = 8', not np.array_equal(fs.World(1, dict(cfg, s=4)).concept_post, orig))

print('\nALL CHECKS PASSED' if not FAILED else f'\n{len(FAILED)} CHECK(S) FAILED: {FAILED}')
sys.exit(1 if FAILED else 0)
