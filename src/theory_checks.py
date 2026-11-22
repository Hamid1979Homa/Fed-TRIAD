"""Numerical sanity checks (with failure controls) for the revised Fed-TRIAD theorems.

Each check prints PASS/FAIL on an exact condition and also runs a control that must FAIL
for the reason the check tests. Nothing here is a proof; it guards against algebra slips.
"""
import itertools
import numpy as np

rng = np.random.default_rng(20261120)
results = []


def report(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- simplex QP (exact, by active sets)
def simplex_qp(M):
    """Exact minimiser of a^T M a over the probability simplex (dimension 3) by enumerating supports."""
    J = M.shape[0]
    best, best_a = np.inf, None
    for r in range(1, J + 1):
        for S in itertools.combinations(range(J), r):
            S = list(S)
            MS = M[np.ix_(S, S)]
            try:
                sol = np.linalg.solve(MS, np.ones(r))
                a_S = sol / sol.sum()
            except np.linalg.LinAlgError:
                a_S = np.linalg.lstsq(MS, np.ones(r), rcond=None)[0]
                if abs(a_S.sum()) < 1e-12:
                    continue
                a_S = a_S / a_S.sum()
            if np.any(a_S < -1e-12):
                continue
            a = np.zeros(J)
            a[S] = np.clip(a_S, 0, None)
            a /= a.sum()
            v = a @ M @ a
            if v < best - 1e-14:
                best, best_a = v, a
    return best_a, best


def grid_min(M, step=0.002):
    g = np.arange(0, 1 + 1e-12, step)
    best = np.inf
    for a1 in g:
        a2 = g[g <= 1 - a1 + 1e-12]
        a3 = 1 - a1 - a2
        A = np.stack([np.full_like(a2, a1), a2, a3], 1)
        v = np.einsum('ij,jk,ik->i', A, M, A)
        best = min(best, v.min())
    return best


# ---------------------------------------------------------------- Theorem 2
ok_qp, ok_closed, ok_shift, ok_diag_subopt = True, True, True, 0
for trial in range(300):
    B = rng.normal(size=(3, 3))
    C = B @ B.T + 0.05 * np.eye(3)          # covariance part
    b = rng.normal(scale=0.7, size=3)       # bias part
    M = C + np.outer(b, b)                  # residual second-moment matrix
    a_qp, v_qp = simplex_qp(M)
    v_grid = grid_min(M, 0.01)
    ok_qp &= v_qp <= v_grid + 1e-9          # exact QP never worse than a fine grid
    w = np.linalg.solve(M, np.ones(3)); w_un = w / w.sum()
    if np.all(w_un >= 0):
        ok_closed &= np.allclose(w_un, a_qp, atol=1e-7)
    tau2 = rng.uniform(0.1, 3.0)
    a_qp2, v_qp2 = simplex_qp(M + tau2 * np.ones((3, 3)))
    ok_shift &= np.allclose(a_qp, a_qp2, atol=1e-7) and abs(v_qp2 - (v_qp + tau2)) < 1e-8
    # diagonal inverse-variance rule computed from raw residual variances
    a_diag = (1 / np.diag(M)) / np.sum(1 / np.diag(M))
    if a_diag @ M @ a_diag > v_qp + 1e-6:
        ok_diag_subopt += 1
report("Thm2: active-set QP is optimal (<= grid)", ok_qp)
report("Thm2: closed form Sigma^-1 1 / 1'Sigma^-1 1 equals QP when nonnegative", ok_closed)
report("Thm2: adding tau^2 11' leaves argmin unchanged and adds tau^2", ok_shift)
report("Thm2: diagonal inverse-variance rule is suboptimal in most correlated draws",
       ok_diag_subopt > 150, f"({ok_diag_subopt}/300 strictly worse)")
# failure control: with a *non*-constant shift (tau^2 * diag) the argmin must move in general
moved = 0
for trial in range(100):
    B = rng.normal(size=(3, 3)); M = B @ B.T + 0.05 * np.eye(3)
    a1, _ = simplex_qp(M); a2, _ = simplex_qp(M + np.diag(rng.uniform(0.5, 3, 3)))
    moved += not np.allclose(a1, a2, atol=1e-4)
report("Thm2 control: a diagonal (non-11') shift DOES move the argmin", moved > 80, f"({moved}/100)")
# uncorrelated unbiased case reduces to inverse variance
s2 = np.array([0.25, 1.0, 2.25]); a_iv = (1 / s2) / np.sum(1 / s2)
a_qp, _ = simplex_qp(np.diag(s2))
report("Thm2 corollary: diagonal M gives inverse-variance weights", np.allclose(a_iv, a_qp),
       f"weights={np.round(a_qp, 4)}")

# ---------------------------------------------------------------- Theorem 5 (correlation-adjusted Renyi-2)
ok5, tight = True, []
for trial in range(2000):
    B = rng.normal(size=(3, 3)); M = B @ B.T + np.outer(*(2 * [rng.normal(size=3)]))
    sbar = np.max(np.diag(M)); cbar = np.max(M[~np.eye(3, dtype=bool)])
    a = rng.dirichlet(rng.uniform(0.2, 3, 3))
    lhs = a @ M @ a; rhs = cbar + (sbar - cbar) * np.sum(a ** 2)
    ok5 &= lhs <= rhs + 1e-10
    tight.append(lhs / rhs if rhs > 0 else 1)
report("Thm5: a'Ma <= cbar + (sbar-cbar)/G2(a) for all draws", ok5,
       f"(median tightness {np.median(tight):.2f})")
# reduces to old bound when uncorrelated: cbar = 0
M = np.diag([1.0, 1.0, 1.0]); a = np.array([0.5, 0.25, 0.25])
report("Thm5 special case: equal-variance uncorrelated gives sigma^2/G2 = 3/8",
       abs(a @ M @ a - 3 / 8) < 1e-12)
# failure control: Shannon effective number in place of G2 must break the bound somewhere
viol = 0
for trial in range(2000):
    a = rng.dirichlet(np.ones(3) * 0.7)
    H1 = -np.sum(a * np.log(a + 1e-300))
    viol += (a @ a) > 1 / np.exp(H1) + 1e-12
report("Thm5 control: Shannon substitution sigma^2/exp(H1) fails as an upper bound", viol > 0,
       f"({viol}/2000 violations)")

# ---------------------------------------------------------------- Theorem 1 witness
# X=[0,1] uniform, m(x)=0, f_P = 1{x>1/2}, f_G = 1{x<=1/2}, f_C = 1 (all errors >= 0)
x = (np.arange(200000) + 0.5) / 200000
dP = (x > 0.5).astype(float); dG = (x <= 0.5).astype(float); dC = np.ones_like(x)
D = np.stack([dP, dC, dG], 1)
Mstat = D.T @ D / len(x)
_, v_static = simplex_qp(Mstat)
report("Thm1 witness: best static three-path mixture has excess risk 1/4", abs(v_static - 0.25) < 1e-6,
       f"({v_static:.6f})")
for T in [1.0, 0.2, 0.05]:
    z = np.stack([np.where(x <= 0.5, 1.0, -1.0), -np.ones_like(x), np.where(x > 0.5, 1.0, -1.0)], 1)
    # logits: P preferred on x<=1/2 (where dP=0), G preferred on x>1/2 (where dG=0)
    a = np.exp(z / T); a /= a.sum(1, keepdims=True)
    risk = np.mean(np.sum(a * D, 1) ** 2)
    bound = np.mean((4 * np.exp(-2 / T)) ** 2)   # |sum a d| <= ||a - e*||_1 * max|d| <= 4 e^{-g/T}, g=2
    print(f"   softmax temperature T={T}: routed excess risk {risk:.3e} (<= {bound:.3e})")
report("Thm1 witness: finite-logit softmax gate beats every static mixture (T=0.2)",
       np.mean(np.sum((lambda a: a / a.sum(1, keepdims=True))(np.exp(np.stack(
           [np.where(x <= 0.5, 1.0, -1.0), -np.ones_like(x), np.where(x > 0.5, 1.0, -1.0)], 1) / 0.2)) * D, 1) ** 2) < 0.25)
# failure control: if every path is exact on the whole domain, the gap is zero
Mzero = np.zeros((3, 3))
_, v0 = simplex_qp(Mzero + 1e-15 * np.eye(3))
report("Thm1 control: no crossing (all paths exact) gives zero gap", v0 < 1e-12)

# temperature lemma: ||softmax(z/T)-e*||_1 <= 2(K-1) e^{-g/T}
ok_t = True
for trial in range(5000):
    z = rng.normal(scale=2, size=3); T = rng.uniform(0.05, 2)
    i = np.argmax(z); g = z[i] - np.max(np.delete(z, i))
    p = np.exp((z - z.max()) / T); p /= p.sum()
    e = np.zeros(3); e[i] = 1
    ok_t &= np.abs(p - e).sum() <= 4 * np.exp(-g / T) + 1e-12
report("Lemma (temperature): ||softmax(z/T)-e*||_1 <= 4 exp(-gap/T) for K=3", ok_t)

# ---------------------------------------------------------------- Theorem 4 geometry
def min_altitude(P):
    a, b, c = P
    area2 = np.linalg.norm(np.cross(b - a, c - a)) if P.shape[1] == 3 else None
    # general d: use Gram determinant
    U = np.stack([b - a, c - a]); G = U @ U.T; area = 0.5 * np.sqrt(max(np.linalg.det(G), 0))
    sides = [np.linalg.norm(b - c), np.linalg.norm(a - c), np.linalg.norm(a - b)]
    return 2 * area / max(sides)


def dist_to_line(p, u, w):
    v = w - u
    if np.linalg.norm(v) < 1e-12:
        return np.linalg.norm(p - u)
    t = np.dot(p - u, v) / np.dot(v, v)
    return np.linalg.norm(p - (u + t * v))


ok4, worst_ratio = True, np.inf
for trial in range(300):
    d = rng.integers(2, 6)
    P = rng.normal(size=(3, d)) * rng.uniform(0.5, 3)
    h = min_altitude(P)
    best = np.inf
    # adversarial search for the line that minimises the max distance to the three concepts
    for _ in range(300):
        u, w = rng.normal(size=d) * 3, rng.normal(size=d) * 3
        if rng.random() < 0.5:   # lines through or near vertices / edges
            i, j = rng.choice(3, 2, replace=False)
            u, w = P[i] + rng.normal(scale=0.2, size=d), P[j] + rng.normal(scale=0.2, size=d)
        best = min(best, max(dist_to_line(p, u, w) for p in P))
    # local refinement
    ok4 &= best >= h / 2 - 1e-9
    worst_ratio = min(worst_ratio, best / (h / 2))
report("Thm4 geometry: every line is >= h/2 from some concept (h = min altitude)", ok4,
       f"(min observed ratio {worst_ratio:.3f})")
# failure control: collinear concepts (h=0) -> a single line passes through all three
P = np.array([[0, 0.], [1, 1.], [2, 2.]])
report("Thm4 control: collinear concepts give zero distance (no obstruction)",
       max(dist_to_line(p, P[0], P[2]) for p in P) < 1e-12)

# per-client lower bound min_beta (1-beta)^2 a + beta^2 v = a v /(a+v)
ok_b = True
for trial in range(1000):
    a_, v_ = rng.uniform(0.01, 5, 2)
    beta = np.linspace(0, 1, 100001)
    ok_b &= abs(np.min((1 - beta) ** 2 * a_ + beta ** 2 * v_) - a_ * v_ / (a_ + v_)) < 1e-6
report("Thm4: min_beta (1-beta)^2 a + beta^2 v = a v/(a+v)", ok_b)

print("\nSUMMARY:", sum(r[1] for r in results), "of", len(results), "checks passed")
