"""End-to-end numerical study of Fed-TRIAD (revised Algorithms 1-3).

Decoupled path training (personal / cluster / global trained on their own squared losses),
an input-dependent softmax gate trained on a held-out split with the paths frozen,
k-means + silhouette reclustering with hysteresis, Hungarian relabelling and
cluster-model transfer, and a stale-client rule. Multi-output (q=3) linear-Gaussian data
with a persistent concept shift at round tau that puts three concepts into one static cluster.

Usage: python fedtriad_sim.py --seeds 1 2 3 4 5 --m 12 --out results_m12.npy
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

CFG = dict(
    N=30, d=8, q=3, T=200, tau=100, n_path=60, n_gate=30, n_test=2000,
    noise=0.5, kappa=0.6, omega=0.6,
    E=5, batch=16, lr=0.05, ridge=1e-3, lr_g=0.5, E_g=10, reg_g=1e-2, eta=1.0,
    t0=20, W=10, A_max=20, tau_imp=0.02, K_max=8,
    s=8,                       # number of clients that change concept at tau (first half -> 2, rest -> new 4)
    ditto_lams=(0.1, 1.0, 10.0),
    match_cluster_batches=False,
)

VARIANTS = {
    # name: (paths used [P, C, G], cluster regime, gate type)
    'Local':          ((1, 0, 0), None,       'fixed'),
    'FedAvg':         ((0, 0, 1), None,       'fixed'),
    'CFL-static':     ((0, 1, 0), 'static',   'fixed'),
    'CFL-adaptive':   ((0, 1, 0), 'adaptive', 'fixed'),
    'PG-const':       ((1, 0, 1), None,       'const'),   # APFL-style per-client scalar mixture
    'PG-gate':        ((1, 0, 1), None,       'input'),   # Flow-style per-input local/global
    'PC-gate':        ((1, 1, 0), 'adaptive', 'input'),   # two-path local/cluster gate
    'TRIAD-uniform':  ((1, 1, 1), 'adaptive', 'fixed'),
    'TRIAD-const':    ((1, 1, 1), 'adaptive', 'const'),
    'TRIAD-static':   ((1, 1, 1), 'static',   'input'),
    'TRIAD':          ((1, 1, 1), 'adaptive', 'input'),
    'TRIAD-oracle':   ((1, 1, 1), 'oracle',   'input'),   # true partition known (comparator)
}


def feat(X):
    """psi(x) = [x, x * 1{x_1 > 0}] (piecewise-linear features shared by all paths)."""
    return np.concatenate([X, X * (X[:, :1] > 0)], 1)


def gate_feat(X):
    """Gate information s(x) = [1, 1{x_1>0}, x_1]."""
    return np.stack([np.ones(len(X)), (X[:, 0] > 0).astype(float), X[:, 0]], 1)


class World:
    def __init__(self, seed, cfg):
        c = cfg
        self.c = c
        rng = np.random.default_rng(seed)
        self.rng = rng
        p = 2 * c['d']
        self.p = p
        Theta0 = rng.normal(size=(c['q'], p))
        self.Theta = {k: Theta0 + c['kappa'] * rng.normal(size=(c['q'], p)) for k in (1, 2, 3, 4)}
        self.Phi = [np.concatenate([np.zeros((c['q'], c['d'])),
                                    c['omega'] * rng.normal(size=(c['q'], c['d']))], 1)
                    for _ in range(c['N'])]
        N = c['N']
        self.concept_pre = np.array([1] * 12 + [2] * 12 + [3] * 6)
        self.concept_post = self.concept_pre.copy()
        h = (c['s'] + 1) // 2                      # s=8 -> clients 0-3 to concept 2, 4-7 to concept 4
        self.concept_post[0:h] = 2
        self.concept_post[h:c['s']] = 4
        self.phase_data = {}
        for phase, conc in (('pre', self.concept_pre), ('post', self.concept_post)):
            D = []
            for i in range(N):
                B = self.Theta[conc[i]] + self.Phi[i]
                def draw(n):
                    X = rng.normal(size=(n, c['d']))
                    F = feat(X)
                    M = F @ B.T
                    return X, F, M, M + c['noise'] * rng.normal(size=M.shape)
                Xp, Fp, _, Yp = draw(c['n_path'])
                Xg, Fg, _, Yg = draw(c['n_gate'])
                Xt, Ft, Mt, _ = draw(c['n_test'])
                D.append(dict(Fp=Fp, Yp=Yp, Fg=Fg, Sg=gate_feat(Xg), Yg=Yg,
                              Ft=Ft, St=gate_feat(Xt), Mt=Mt))
            self.phase_data[phase] = D

    def data(self, t, i):
        return self.phase_data['pre' if t < self.c['tau'] else 'post'][i]

    def true_partition(self, t):
        conc = self.concept_pre if t < self.c['tau'] else self.concept_post
        return conc.copy()


def ditto_sgd(V0, Gm, F, Y, c, rng, lam):
    """Ditto personal update: E SGD steps on 0.5*||V f - y||^2 + 0.5*ridge*||V||^2 + 0.5*lam*||V - G||^2."""
    V = V0.copy()
    for _ in range(c['E']):
        idx = rng.choice(len(F), size=min(c['batch'], len(F)), replace=False)
        Fb, Yb = F[idx], Y[idx]
        Gr = (Fb @ V.T - Yb).T @ Fb / len(idx) + c['ridge'] * V + lam * (V - Gm)
        V -= c['lr'] * Gr
    return V


def local_sgd(W0, F, Y, c, rng, steps=None, *, batches=None):
    """E steps of minibatch SGD on 0.5*||W f - y||^2 + 0.5*ridge*||W||^2."""
    W = W0.copy()
    for step in range(c['E'] if steps is None else steps):
        idx = rng.choice(len(F), size=min(c['batch'], len(F)), replace=False)
        if batches is not None:
            # Keep the original random stream on the matched-minibatch control.
            idx = batches[step]
        Fb, Yb = F[idx], Y[idx]
        G = (Fb @ W.T - Yb).T @ Fb / len(idx) + c['ridge'] * W
        W -= c['lr'] * G
    return W


class ClusterRegime:
    """Cluster assignments + cluster models + reclustering/transfer bookkeeping."""

    def __init__(self, kind, N, init_model):
        self.kind = kind
        self.labels = np.zeros(N, dtype=int)         # everyone in cluster 0 before t0
        self.models = {0: init_model.copy()}
        self.n_reclusters = 0
        self.history = []

    def relabel_and_transfer(self, new_labels):
        """Hungarian matching on membership overlap; matched clusters inherit models,
        unmatched new clusters copy the model of their max-overlap old cluster."""
        old = self.labels
        old_ids, new_ids = np.unique(old), np.unique(new_labels)
        O = np.array([[np.sum((old == a) & (new_labels == b)) for b in new_ids] for a in old_ids])
        r, cidx = linear_sum_assignment(-O)
        mapping = {}
        for a_i, b_i in zip(r, cidx):
            if O[a_i, b_i] > 0:
                mapping[new_ids[b_i]] = old_ids[a_i]
        next_id = max(max(self.models) + 1, 0)
        new_models, final = {}, np.empty_like(new_labels)
        for b_i, b in enumerate(new_ids):
            if b in mapping:
                lab = mapping[b]
                new_models[lab] = self.models[lab].copy()
            else:
                lab = next_id; next_id += 1
                src = old_ids[np.argmax(O[:, b_i])]
                new_models[lab] = self.models[src].copy()
            final[new_labels == b] = lab
        self.labels = final
        self.models = new_models


def kmeans_labels(Z, K, seed):
    return KMeans(n_clusters=K, init='k-means++', n_init=10, algorithm='lloyd',
                  max_iter=300, tol=1e-4, random_state=seed).fit_predict(Z)


def algorithm3(Z, cur_labels, K_cur, cfg, seed, initial=False):
    """Adaptive cluster reassignment: candidates K-1, K, K+1 (or 2..6 at initialisation);
    quality = silhouette; accept only if it beats the current partition by tau_imp."""
    if len(Z) < 3:
        return cur_labels, -np.inf, False
    candidates = range(2, 7) if initial else (K_cur - 1, K_cur, K_cur + 1)
    upper = min(cfg['K_max'], len(Z) - 1, len(np.unique(Z, axis=0)))
    cur_count = len(np.unique(cur_labels))
    q_cur = silhouette_score(Z, cur_labels) if not initial and 1 < cur_count < len(Z) else -np.inf
    best = (-np.inf, None)
    for K in candidates:
        if not 2 <= K <= upper:
            continue
        lab = kmeans_labels(Z, K, seed)
        if not 1 < len(np.unique(lab)) < len(Z):
            continue
        q = silhouette_score(Z, lab)
        if q > best[0]:
            best = (q, lab)
    if best[0] > q_cur + (0 if initial else cfg['tau_imp']) and (
            initial or adjusted_rand_score(cur_labels, best[1]) < 1.0):
        return best[1], best[0], True
    return cur_labels, q_cur, False


def softmax_masked(Z, mask):
    Z = np.where(mask[None, :] > 0, Z, -np.inf)
    Z = Z - Z.max(1, keepdims=True)
    A = np.exp(Z) * mask[None, :]
    return A / A.sum(1, keepdims=True)


def gate_weights(gtype, params, S, mask):
    if gtype == 'fixed':
        return np.tile(mask / mask.sum(), (len(S), 1))
    if gtype == 'const':
        return softmax_masked(np.tile(params, (len(S), 1)), mask)
    return softmax_masked(S @ params.T, mask)        # params: 3 x g


def gate_step(gtype, params, S, Fpaths, Y, mask, cfg):
    """GD on mean squared routed loss plus (reg_g/2)*||params||^2."""
    for _ in range(cfg['E_g']):
        A = gate_weights(gtype, params, S, mask)                   # n x 3
        R = np.einsum('nj,njq->nq', A, Fpaths) - Y                 # n x q
        g = 2 * np.einsum('nq,njq->nj', R, Fpaths)                 # dL/da_j
        dz = A * (g - np.sum(A * g, 1, keepdims=True))             # softmax jacobian
        dz *= mask[None, :]
        if gtype == 'const':
            params = params - cfg['lr_g'] * (dz.mean(0) + cfg['reg_g'] * params)
        else:
            params = params - cfg['lr_g'] * (dz.T @ S / len(S) + cfg['reg_g'] * params)
    return params


def run(seed, cfg, m):
    c = dict(CFG, **cfg)
    validate_config(c, m, [seed])
    world = World(seed, c)
    rng = np.random.default_rng(10_000 + seed)
    N, q, p = c['N'], c['q'], world.p
    P = [np.zeros((q, p)) for _ in range(N)]
    last_seen = np.full(N, -10 ** 9)
    G = np.zeros((q, p))
    regimes = {k: ClusterRegime(k, N, G) for k in ('static', 'adaptive', 'oracle')}
    regimes['oracle'].labels = world.true_partition(0) - 1
    regimes['oracle'].models = {k: G.copy() for k in np.unique(regimes['oracle'].labels)}
    gates = {v: [None] * N for v in VARIANTS}
    for v, (mask, reg, gt) in VARIANTS.items():
        for i in range(N):
            gates[v][i] = np.zeros(3) if gt == 'const' else np.zeros((3, 3))
    masks = {v: np.array(mask, float) for v, (mask, _, _) in VARIANTS.items()}
    evaluation_weights = {v: [None] * N for v in VARIANTS}
    cluster_rng = np.random.default_rng(0) if c['match_cluster_batches'] else None
    T = c['T']
    names = list(VARIANTS) + ['Ditto']
    excess = {v: np.zeros(T) for v in names}
    worst_client = {v: np.zeros(T) for v in names}
    # Ditto: own RNG so that the random streams of all other methods are unchanged
    rng_d = np.random.default_rng(20_000 + seed)
    lams = list(c['ditto_lams'])
    Vd = {l: [np.zeros((q, p)) for _ in range(N)] for l in lams}
    lam_sel = np.full(N, len(lams) // 2)           # index of the selected lambda per client
    lam_hist = []
    ari_adapt, ari_static, K_adapt, cta_round = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    recluster_rounds = []

    for t in range(T):
        # ---------- oracle regime: instant re-partition at tau (new cluster starts from global)
        if t == c['tau']:
            for weights in evaluation_weights.values():
                weights[:] = [None] * N
            reg = regimes['oracle']
            new = world.true_partition(t) - 1
            for k in np.unique(new):
                if k not in reg.models:
                    reg.models[k] = G.copy()
            reg.labels = new
        S_t = rng.choice(N, size=m, replace=False)
        # ---------- client updates (Algorithm 2), decoupled paths
        dG = []
        dC = {k: {} for k in regimes}
        for i in S_t:
            D = world.data(t, i)
            P[i] = local_sgd(P[i], D['Fp'], D['Yp'], c, rng)
            last_seen[i] = t
            dG.append(local_sgd(G, D['Fp'], D['Yp'], c, rng) - G)
            batches = None
            if cluster_rng is not None:
                cluster_rng.bit_generator.state = rng.bit_generator.state
                n = len(D['Fp'])
                batches = [cluster_rng.choice(n, size=min(c['batch'], n), replace=False)
                           for _ in range(c['E'])]
            for k, reg in regimes.items():
                lab = reg.labels[i]
                dC[k].setdefault(lab, []).append(
                    local_sgd(reg.models[lab], D['Fp'], D['Yp'], c, rng,
                              batches=batches) - reg.models[lab])
            # Ditto personal models (received global G), lambda chosen on the held-out gate split
            hold = []
            for l in lams:
                Vd[l][i] = ditto_sgd(Vd[l][i], G, D['Fp'], D['Yp'], c, rng_d, l)
                hold.append(np.mean(np.sum((D['Fg'] @ Vd[l][i].T - D['Yg']) ** 2, 1)))
            lam_sel[i] = int(np.argmin(hold))
        lam_hist.append(np.bincount(lam_sel, minlength=len(lams)))
        # gates are trained with the paths the client holds this round (received shared + new personal)
        for i in S_t:
            D = world.data(t, i)
            Fg = D['Fg']
            for v, (mask, rname, gt) in VARIANTS.items():
                if gt == 'fixed':
                    continue
                Cm = regimes[rname].models[regimes[rname].labels[i]] if rname else np.zeros((q, p))
                Fpaths = np.stack([Fg @ P[i].T, Fg @ Cm.T, Fg @ G.T], 1)
                gates[v][i] = gate_step(gt, gates[v][i], D['Sg'], Fpaths, D['Yg'],
                                        masks[v], c)
                evaluation_weights[v][i] = None
        # The sample average estimates the full-population mean of the local deltas.
        G = G + c['eta'] * np.mean(dG, 0)
        for k, reg in regimes.items():
            for lab, ds in dC[k].items():
                reg.models[lab] = reg.models[lab] + c['eta'] * np.mean(ds, 0)
        # ---------- reclustering (Algorithm 3)
        cta = 0.0
        if t == c['t0'] or (t > c['t0'] and (t - c['t0']) % c['W'] == 0):
            fresh = np.where(t - last_seen <= c['A_max'])[0]
            Z = np.stack([P[i].ravel() for i in fresh])
            for rname in ('static', 'adaptive'):
                reg = regimes[rname]
                if rname == 'static' and t > c['t0']:
                    continue
                cur = reg.labels[fresh]
                new_f, qsc, changed = algorithm3(Z, cur, len(np.unique(reg.labels)), c, seed,
                                                 initial=(t == c['t0']))
                if changed:
                    prev = reg.labels.copy()
                    full = reg.labels.copy()
                    # stale clients keep their lineage: offset fresh labels to avoid collisions
                    off = 10 ** 6
                    full[fresh] = new_f + off
                    stale = np.setdiff1d(np.arange(N), fresh)
                    for i in stale:   # lineage rule: follow the max-overlap new cluster of the old label
                        same = fresh[prev[fresh] == prev[i]]
                        if len(same):
                            vals, cnt = np.unique(full[same], return_counts=True)
                            full[i] = vals[np.argmax(cnt)]
                    if rname == 'static':
                        reg.labels = np.zeros(N, dtype=int)
                    reg.relabel_and_transfer(full)
                    if rname == 'adaptive' and t > c['t0']:
                        reg.n_reclusters += 1
                        recluster_rounds.append(t)
                        cta = 1 - adjusted_rand_score(prev, reg.labels)
        cta_round[t] = cta
        truth = world.true_partition(t)
        ari_adapt[t] = adjusted_rand_score(truth, regimes['adaptive'].labels)
        ari_static[t] = adjusted_rand_score(truth, regimes['static'].labels)
        K_adapt[t] = len(np.unique(regimes['adaptive'].labels))
        # Estimate excess risk on held-out inputs drawn once per client and phase.
        per_client = {v: np.zeros(N) for v in names}
        for i in range(N):
            D = world.data(t, i)
            Ft = D['Ft']
            base = {'P': Ft @ P[i].T, 'G': Ft @ G.T}
            Cpred = {k: Ft @ reg.models[reg.labels[i]].T for k, reg in regimes.items()}
            paths = {}
            for v, (mask, rname, gt) in VARIANTS.items():
                Cp = Cpred[rname] if rname else np.zeros_like(base['P'])
                if sum(mask) == 1:
                    pred = (base['P'], Cp, base['G'])[mask.index(1)]
                else:
                    if rname not in paths:
                        paths[rname] = np.stack([base['P'], Cp, base['G']], 1)
                    A = evaluation_weights[v][i]
                    if A is None:
                        A = gate_weights(gt, gates[v][i], D['St'], masks[v])
                        evaluation_weights[v][i] = A
                    pred = np.einsum('nj,njq->nq', A, paths[rname])
                per_client[v][i] = np.mean(np.sum((pred - D['Mt']) ** 2, 1))
            Vsel = Vd[lams[lam_sel[i]]][i]
            per_client['Ditto'][i] = np.mean(np.sum((Ft @ Vsel.T - D['Mt']) ** 2, 1))
        for v in names:
            excess[v][t] = per_client[v].mean()
            worst_client[v][t] = per_client[v].max()

    # ---------- Theorem 2 / Theorem 5 diagnostics on the final TRIAD paths (fresh evaluation data)
    diag = theorem_diagnostics(world, P, G, regimes['adaptive'], gates['TRIAD'], c, rng)
    if any(not np.isfinite(curve).all() for curve in excess.values()):
        raise ValueError('The run produced a non-finite risk. Check the step sizes and data scales.')
    out = dict(excess=excess, worst=worst_client, ari_adapt=ari_adapt, ari_static=ari_static,
               K_adapt=K_adapt, cta_round=cta_round, recluster_rounds=recluster_rounds,
               n_reclusters=regimes['adaptive'].n_reclusters,
               static_K=len(np.unique(regimes['static'].labels)), diag=diag,
               ditto_lambda_counts=np.array(lam_hist))
    return out


def simplex_qp(M):
    import itertools
    best, best_a = np.inf, None
    for r in range(1, 4):
        for S in itertools.combinations(range(3), r):
            S = list(S)
            MS = M[np.ix_(S, S)]
            try:
                sol = np.linalg.solve(MS + 1e-12 * np.eye(r), np.ones(r))
            except np.linalg.LinAlgError:
                continue
            if abs(sol.sum()) < 1e-15:
                continue
            aS = sol / sol.sum()
            if np.any(aS < -1e-12):
                continue
            a = np.zeros(3); a[S] = np.clip(aS, 0, None); a /= a.sum()
            v = a @ M @ a
            if v < best:
                best, best_a = v, a
    return best_a


def theorem_diagnostics(world, P, G, reg, gate_params, c, rng, n_fit=4000, n_eval=4000, nbins=8):
    """Per client: bin s(x) by x_1 quantiles; estimate residual second-moment matrices M_b
    on a fit sample; build (i) QP-optimal (Thm 2), (ii) diagonal inverse-variance, (iii) uniform,
    (iv) best-single-path bin-constant gates; evaluate all and the learned gate on an
    independent sample. Also the Thm 5 pointwise bound with M(x) = d d^T + s^2 q 11^T."""
    res = {k: [] for k in ('learned', 'qp', 'diag', 'uniform', 'best_single',
                           'thm5_ratio', 'thm5_holds', 'mean_offdiag_corr')}
    edges = np.quantile(rng.normal(size=100000), np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    for i in range(c['N']):
        B = world.Theta[world.concept_post[i]] + world.Phi[i]
        Cm = reg.models[reg.labels[i]]
        def sample(n):
            X = rng.normal(size=(n, c['d'])); F = feat(X); M = F @ B.T
            Y = M + c['noise'] * rng.normal(size=M.shape)
            Fp = np.stack([F @ P[i].T, F @ Cm.T, F @ G.T], 1)
            return X, Fp, M, Y
        Xf, Ff, Mf, Yf = sample(n_fit)
        Xe, Fe, Me, Ye = sample(n_eval)
        bf, be = np.digitize(Xf[:, 0], edges) - 1, np.digitize(Xe[:, 0], edges) - 1
        Rf = Ff - Yf[:, None, :]
        rules = {k: np.zeros((nbins, 3)) for k in ('qp', 'diag', 'uniform', 'best_single')}
        corrs = []
        for b in range(nbins):
            Rb = Rf[bf == b]
            Mb = np.einsum('njq,nkq->jk', Rb, Rb) / len(Rb)
            rules['qp'][b] = simplex_qp(Mb)
            diag = np.diag(Mb)
            iv = (diag == 0).astype(float) if np.any(diag == 0) else 1 / diag
            rules['diag'][b] = iv / iv.sum()
            rules['uniform'][b] = np.ones(3) / 3
            e = np.zeros(3); e[np.argmin(np.diag(Mb))] = 1; rules['best_single'][b] = e
            sd = np.sqrt(diag)
            scale = np.outer(sd, sd)
            Cr = np.divide(Mb, scale, out=np.zeros_like(Mb), where=scale > 0)
            corrs.append(Cr[~np.eye(3, dtype=bool)].mean())
        for k, A in rules.items():
            pred = np.einsum('nj,njq->nq', A[be], Fe)
            res[k].append(np.mean(np.sum((pred - Me) ** 2, 1)))
        Ag = gate_weights('input', gate_params[i], gate_feat(Xe), np.ones(3))
        pred = np.einsum('nj,njq->nq', Ag, Fe)
        res['learned'].append(np.mean(np.sum((pred - Me) ** 2, 1)))
        # Theorem 5 (pointwise form, s = x): M(x)_{jk} = <d_j,d_k> + q s^2
        d = Fe - Me[:, None, :]
        Mx = np.einsum('njq,nkq->njk', d, d) + q_noise(c)
        diag_max = np.max(np.diagonal(Mx, axis1=1, axis2=2), 1)
        off = Mx.copy(); off[:, np.arange(3), np.arange(3)] = -np.inf
        off_max = off.reshape(len(Mx), -1).max(1)
        g2 = np.sum(Ag ** 2, 1)
        lhs = np.einsum('nj,njk,nk->n', Ag, Mx, Ag)
        rhs = off_max + (diag_max - off_max) * g2
        res['thm5_holds'].append(bool(np.all(lhs <= rhs + 1e-9)))
        res['thm5_ratio'].append(float(lhs.mean() / rhs.mean()) if rhs.mean() else 0.0)
        res['mean_offdiag_corr'].append(float(np.mean(corrs)))
    return {k: np.array(v) for k, v in res.items()}


def scenario_overrides(name):
    return dict(n_path=25, n_gate=15, noise=1.0, kappa=0.3, omega=0.3) if name == 'scarce' else {}


def q_noise(c):
    return c['q'] * c['noise'] ** 2


def parse_overrides(values):
    overrides = {}
    for item in values:
        key, equal, value = item.partition('=')
        if not equal or key not in CFG or key in overrides:
            raise ValueError(f'Use each known parameter once as name=value: {item!r}')
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{key} requires a JSON number, list, or boolean') from exc
    return overrides


def validate_config(cfg, m, seeds):
    for key, default in CFG.items():
        value = cfg[key]
        if type(default) is int and type(value) is not int:
            raise ValueError(f'{key} must be an integer')
        if type(default) is bool and type(value) is not bool:
            raise ValueError(f'{key} must be true or false')
        if type(default) is float and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(f'{key} must be a finite nonnegative number')
    if cfg['N'] != 30 or not 1 <= m <= cfg['N']:
        raise ValueError('This data generator uses 30 clients. Set m between 1 and 30.')
    if not (0 <= cfg['tau'] < cfg['T'] and 0 <= cfg['t0'] < cfg['T']):
        raise ValueError('Require 0 <= tau,t0 < T')
    for key in ('d', 'q', 'n_path', 'n_gate', 'n_test', 'batch', 'W', 'E'):
        if cfg[key] < 1:
            raise ValueError(f'{key} must be at least 1')
    if not 0 <= cfg['s'] <= 12 or min(cfg['E_g'], cfg['A_max']) < 0 or cfg['K_max'] < 2:
        raise ValueError('Require 0 <= s <= 12, E_g and A_max >= 0, and K_max >= 2')
    lams = cfg['ditto_lams']
    if (not isinstance(lams, (list, tuple)) or not lams
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in lams)
            or len(set(lams)) != len(lams)):
        raise ValueError('ditto_lams must be a nonempty list of distinct, finite nonnegative numbers')
    if len(set(seeds)) != len(seeds) or not all(0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Use distinct integer seeds in [0, 2**32)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--m', type=int, default=12)
    ap.add_argument('--out', default='results.npy', type=Path)
    ap.add_argument('--scenario', default='drift', choices=['drift', 'scarce'])
    ap.add_argument('--set', nargs='+', default=[], help='e.g. kappa=1.0 s=4 match_cluster_batches=true')
    a = ap.parse_args()
    cfg = dict(CFG)
    try:
        cfg.update(scenario_overrides(a.scenario))
        cfg.update(parse_overrides(a.set))
        validate_config(cfg, a.m, a.seeds)
    except ValueError as exc:
        ap.error(str(exc))
    out = a.out if a.out.suffix == '.npy' else Path(str(a.out) + '.npy')
    config = out.with_suffix('.cfg.json')
    if out.exists() or config.exists():
        ap.error(f'{out} or its configuration already exists. Choose a fresh path.')
    out.parent.mkdir(parents=True, exist_ok=True)
    allres = {}
    for s in a.seeds:
        t0 = time.time()
        r = run(s, cfg, a.m)
        allres[s] = r
        print(f"seed {s}: done in {time.time() - t0:.1f}s; reclusters={r['n_reclusters']} "
              f"static_K={r['static_K']} final_ARI_adapt={r['ari_adapt'][-1]:.3f} "
              f"final_ARI_static={r['ari_static'][-1]:.3f}", flush=True)
        for v in list(VARIANTS) + ['Ditto']:
            print(f"   {v:14s} final excess {r['excess'][v][-20:].mean():.4f}", flush=True)
    np.save(out, allres, allow_pickle=True)
    config.write_text(json.dumps(dict(CFG=cfg, m=a.m, seeds=a.seeds, scenario=a.scenario),
                                 indent=2) + '\n', encoding='utf-8')
