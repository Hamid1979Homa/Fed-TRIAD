"""Summarise the Fed-TRIAD runs into numbers.json + figures. Every number in the manuscript comes from here.
Usage: python analyze.py <results dir> <output dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from paper_tables import write_tables

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
MAIN = ('drift_m12', 'drift_m30', 'scarce_m12')
COMPETITORS = ['Local', 'FedAvg', 'CFL-static', 'CFL-adaptive', 'PG-const', 'PG-gate', 'PC-gate', 'Ditto']
SENS = (['drift_m12'] + [f'kw_k{k}_w{w}' for k in (0.3, 0.6, 1.0) for w in (0.3, 0.6, 1.0) if (k, w) != (0.6, 0.6)]
        + ['s0', 's4', 's12', 'tau0.0', 'tau0.01', 'tau0.05', 'tau0.1', 'W5', 'W20', 'm6', 'drift_m30', 'n30', 'n120'])
runs = {k: np.load(f'{RES}/{k}.npy', allow_pickle=True).item() for k in sorted(set(MAIN) | set(SENS))}
cfgs = {k: json.load(open(f'{RES}/{k}.cfg.json')) for k in runs}
T, tau = cfgs['drift_m12']['CFG']['T'], cfgs['drift_m12']['CFG']['tau']
V = list(next(iter(runs['drift_m12'].values()))['excess'].keys())


def ms(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), n=int(a.size), values=[float(x) for x in a])


def final(R, s, v):
    return R[s]['excess'][v][-20:].mean()


def summarise(R, with_diag=True):
    seeds = sorted(R)
    d = {}
    for v in V:
        fin = [final(R, s, v) for s in seeds]
        worst = [R[s]['worst'][v][-20:].mean() for s in seeds]
        reg = [np.sum(R[s]['excess'][v][tau:] - R[s]['excess']['TRIAD-oracle'][tau:]) for s in seeds]
        slope = [np.polyfit(np.arange(50), np.cumsum(R[s]['excess'][v][tau:] - R[s]['excess']['TRIAD-oracle'][tau:])[-50:], 1)[0]
                 for s in seeds]
        d[v] = dict(final=ms(fin), worst=ms(worst), regret=ms(reg), slope_last50=ms(slope))
    d['_ari_adapt_final'] = ms([R[s]['ari_adapt'][-1] for s in seeds])
    d['_ari_static_final'] = ms([R[s]['ari_static'][-1] for s in seeds])
    d['_J'] = ms([R[s]['n_reclusters'] for s in seeds])
    d['_recluster_rounds'] = {int(s): [int(x) for x in R[s]['recluster_rounds']] for s in seeds}
    d['_first_post_shift_recluster'] = ms([min([x for x in R[s]['recluster_rounds'] if x >= tau] or [np.nan]) for s in seeds])
    d['_static_K'] = [int(R[s]['static_K']) for s in seeds]
    d['_cta_mean'] = ms([R[s]['cta_round'][1:].mean() for s in seeds])
    # excess risk in the round after vs before every accepted re-clustering (Theorem 3 statement)
    jumps = []
    for s in seeds:
        L = R[s]['excess']['TRIAD']
        for t in R[s]['recluster_rounds']:
            if t == tau:                              # the concept shift itself happens in this round; not comparable
                continue
            jumps.append(float(L[t] - L[t - 1]))      # evaluation at round t happens after the change in round t
    d['_recluster_jumps'] = jumps
    lc = np.sum([R[s]['ditto_lambda_counts'][-1] for s in seeds], 0)
    d['_ditto_lambda_share_final'] = [float(x) for x in lc / lc.sum()]
    if with_diag:
        dg = {}
        for k in ('learned', 'qp', 'diag', 'uniform', 'best_single'):
            dg[k] = ms([R[s]['diag'][k].mean() for s in seeds])
            dg[k + '_per_seed'] = [float(R[s]['diag'][k].mean()) for s in seeds]
        dg['thm5_holds_all'] = bool(all(R[s]['diag']['thm5_holds'].all() for s in seeds))
        dg['thm5_ratio'] = ms([R[s]['diag']['thm5_ratio'].mean() for s in seeds])
        dg['resid_corr'] = ms([R[s]['diag']['mean_offdiag_corr'].mean() for s in seeds])
        dg['qp_best_all_seeds'] = bool(all(R[s]['diag']['qp'].mean() <= min(R[s]['diag'][k].mean() for k in
                                                                              ('learned', 'diag', 'uniform', 'best_single'))
                                           for s in seeds))
        dg['diag_worse_than_best_single_all_seeds'] = bool(all(R[s]['diag']['diag'].mean() > R[s]['diag']['best_single'].mean()
                                                               for s in seeds))
        dg['learned_between_qp_and_diag_all_seeds'] = bool(all(R[s]['diag']['qp'].mean() < R[s]['diag']['learned'].mean()
                                                                < R[s]['diag']['diag'].mean() for s in seeds))
        d['_diag'] = dg
    return d


def label(lower, higher, need=8):
    if lower >= need:
        return 'better'
    if higher >= need:
        return 'worse'
    return 'no consistent difference'


def paired(R, a, b):
    """number of seeds where method a has lower final excess risk than method b"""
    return int(sum(final(R, s, a) < final(R, s, b) for s in sorted(R)))


num = {k: summarise(runs[k]) for k in MAIN}
for k in MAIN:   # paired counts: seeds in which Fed-TRIAD has lower final excess risk than each method
    num[k]['_paired_TRIAD_lower'] = {v: paired(runs[k], 'TRIAD', v) for v in V if v != 'TRIAD'}
    num[k]['_paired_const_lower_than_TRIAD'] = paired(runs[k], 'TRIAD-const', 'TRIAD')
    num[k]['_ari_eq1_seeds'] = int(sum(abs(x - 1) < 1e-12 for x in num[k]['_ari_adapt_final']['values']))
sens = {}
for k in SENS:
    R = runs[k]
    smry = num[k] if k in num else summarise(R, with_diag=False)
    best = min(COMPETITORS, key=lambda v: smry[v]['final']['mean'])
    seeds = sorted(R)
    diff = [final(R, s, 'TRIAD') - final(R, s, best) for s in seeds]
    k_best = paired(R, 'TRIAD', best)
    k_static = paired(R, 'TRIAD', 'TRIAD-static')
    k_pc = paired(R, 'TRIAD', 'PC-gate')
    k_ditto = paired(R, 'TRIAD', 'Ditto')
    sens[k] = dict(
        cfg=cfgs[k]['overrides'] if 'overrides' in cfgs[k] else {}, m=cfgs[k]['m'],
        TRIAD=smry['TRIAD']['final'], static=smry['TRIAD-static']['final'], oracle=smry['TRIAD-oracle']['final'],
        PC=smry['PC-gate']['final'], Ditto=smry['Ditto']['final'],
        best_competitor=best, best_value=smry[best]['final'], diff_vs_best=ms(diff),
        seeds_better_than_best=k_best, label_vs_best=label(k_best, paired(R, best, 'TRIAD')),
        seeds_better_than_static=k_static, label_vs_static=label(k_static, paired(R, 'TRIAD-static', 'TRIAD')),
        seeds_better_than_PC=k_pc, label_vs_PC=label(k_pc, paired(R, 'PC-gate', 'TRIAD')),
        seeds_better_than_Ditto=k_ditto, label_vs_Ditto=label(k_ditto, paired(R, 'Ditto', 'TRIAD')),
        J=smry['_J'], ari_adapt=smry['_ari_adapt_final'], ari_static=smry['_ari_static_final'],
        slope_TRIAD=smry['TRIAD']['slope_last50'], slope_static=smry['TRIAD-static']['slope_last50'],
        recluster_jumps=smry['_recluster_jumps'])
num['_sens'] = sens
json.dump(num, open(f'{OUT}/numbers.json', 'w'), indent=1)
write_tables(num, OUT)

# ---------------------------------------------------------------- figures (drift, m=12)
R = runs['drift_m12']; seeds = sorted(R)
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
cols = {'TRIAD': '#1b9e77', 'TRIAD-static': '#d95f02', 'CFL-adaptive': '#7570b3', 'CFL-static': '#e7298a',
        'PC-gate': '#66a61e', 'PG-gate': '#a6761d', 'TRIAD-const': '#666666', 'Local': '#e6ab02',
        'Ditto': '#1f78b4', 'TRIAD-oracle': 'black'}
labels = {'TRIAD': 'Fed-TRIAD (adaptive)', 'TRIAD-static': 'Fed-TRIAD (static partition)',
          'CFL-adaptive': 'Cluster path, adaptive', 'CFL-static': 'Cluster path, static',
          'PC-gate': 'Personal/cluster gate', 'PG-gate': 'Personal/global gate (Flow-style)',
          'TRIAD-const': 'Three paths, constant weights', 'Local': 'Local', 'Ditto': 'Ditto',
          'TRIAD-oracle': 'Oracle partition'}

fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(tau, T)
for v in ('TRIAD-static', 'TRIAD', 'PC-gate', 'PG-gate', 'Ditto', 'TRIAD-const', 'Local'):
    C = np.stack([np.cumsum(R[s]['excess'][v][tau:] - R[s]['excess']['TRIAD-oracle'][tau:]) for s in seeds])
    m, sd = C.mean(0), C.std(0, ddof=1)
    ax.plot(x, m, color=cols[v], label=labels[v], lw=1.6)
    ax.fill_between(x, m - sd, m + sd, color=cols[v], alpha=0.15, lw=0)
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('communication round'); ax.set_ylabel('cumulative regret vs. oracle partition')
ax.legend(fontsize=7.5, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_regret.png', dpi=300); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
for v in ('Local', 'CFL-static', 'CFL-adaptive', 'Ditto', 'TRIAD-static', 'TRIAD', 'TRIAD-oracle'):
    E = np.stack([R[s]['excess'][v] for s in seeds]); m = E.mean(0)
    ax.plot(np.arange(T), m, color=cols[v], label=labels[v], lw=1.3, ls='--' if v == 'TRIAD-oracle' else '-')
ax.set_yscale('log'); ax.axvline(tau, color='k', lw=0.6, ls=':')
ax.set_xlabel('communication round'); ax.set_ylabel('mean excess risk (log scale)')
h_, l_ = ax.get_legend_handles_labels()
ax = axs[1]
A = np.stack([R[s]['ari_adapt'] for s in seeds]); B = np.stack([R[s]['ari_static'] for s in seeds])
ax.plot(np.arange(T), A.mean(0), color=cols['TRIAD'], label='adaptive (Alg. 3)')
ax.plot(np.arange(T), B.mean(0), color=cols['TRIAD-static'], label='static after $t_0$')
ax.axvline(tau, color='k', lw=0.6, ls=':')
ax.set_ylim(-0.05, 1.05); ax.set_xlabel('communication round'); ax.set_ylabel('ARI with true partition')
ax.legend(fontsize=7, frameon=False, loc='lower left')
fig.legend(h_, l_, loc='upper center', ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=[0, 0, 1, 0.86]); fig.savefig(f'{OUT}/fig3_excess_ari.png', dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 2.9))
dg = num['drift_m12']['_diag']
keys = ['qp', 'best_single', 'learned', 'diag', 'uniform']
names = ['QP-optimal\n(Thm 2)', 'best single\npath', 'learned\ngate', 'inverse-variance\n(raw residuals)', 'uniform\n1/3']
ax.bar(range(5), [dg[k]['mean'] for k in keys], yerr=[dg[k]['sd'] for k in keys],
       color=['#1b9e77', '#7570b3', '#66a61e', '#d95f02', '#999999'], capsize=3)
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=8)
ax.set_yscale('log'); ax.set_ylabel('excess risk (held-out)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_gates.png', dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: kappa x omega
ks, ws = (0.3, 0.6, 1.0), (0.3, 0.6, 1.0)
key = lambda k, w: 'drift_m12' if (k, w) == (0.6, 0.6) else f'kw_k{k}_w{w}'
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2))
for ax, (field, cnt, title) in zip(axs, [('diff_vs_best', 'seeds_better_than_best', 'Fed-TRIAD − strongest competitor'),
                                          (None, 'seeds_better_than_static', 'Fed-TRIAD − static partition')]):
    M = np.zeros((3, 3)); txt = [['' for _ in ws] for _ in ks]
    for a, k in enumerate(ks):
        for b, w in enumerate(ws):
            S_ = sens[key(k, w)]
            if field:
                M[a, b] = S_['diff_vs_best']['mean']
                txt[a][b] = f"{M[a, b]:+.3f}\n{S_[cnt]}/10\n{S_['best_competitor']}"
            else:
                Rk = runs[key(k, w)]
                M[a, b] = np.mean([final(Rk, s, 'TRIAD') - final(Rk, s, 'TRIAD-static') for s in sorted(Rk)])
                txt[a][b] = f"{M[a, b]:+.3f}\n{S_[cnt]}/10"
    lim = np.max(np.abs(M))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, txt[a][b], ha='center', va='center', fontsize=6.5,
                    color='white' if abs(M[a, b]) > 0.55 * lim else 'black')
    ax.set_xticks(range(3)); ax.set_xticklabels([str(w) for w in ws]); ax.set_xlabel('client deviation ω')
    ax.set_yticks(range(3)); ax.set_yticklabels([str(k) for k in ks]); ax.set_ylabel('concept separation κ')
    ax.set_title(title, fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(f'{OUT}/fig5_kappa_omega.png', dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig. 6: hysteresis
taus = [0.0, 0.01, 0.02, 0.05, 0.1]
tkey = lambda t: 'drift_m12' if t == 0.02 else f'tau{t}'
fig, ax1 = plt.subplots(figsize=(5.4, 3.0))
Jm = [sens[tkey(t)]['J']['mean'] for t in taus]; Js = [sens[tkey(t)]['J']['sd'] for t in taus]
ax1.errorbar(range(5), Jm, yerr=Js, color='#7570b3', marker='o', capsize=3, label='accepted re-clusterings J')
ax1.set_ylabel('J (after $t_0$)'); ax1.set_xticks(range(5)); ax1.set_xticklabels([str(t) for t in taus])
ax1.set_xlabel(r'hysteresis threshold $\tau_{\mathrm{imp}}$')
ax2 = ax1.twinx()
Em = [sens[tkey(t)]['TRIAD']['mean'] for t in taus]; Es = [sens[tkey(t)]['TRIAD']['sd'] for t in taus]
ax2.errorbar(np.arange(5) + 0.05, Em, yerr=Es, color='#1b9e77', marker='s', capsize=3, label='Fed-TRIAD excess risk')
ax2.set_ylabel('final excess risk')
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(f'{OUT}/fig6_hysteresis.png', dpi=300); plt.close(fig)

for k in SENS:
    S_ = sens[k]
    print(f"{k:14s} TRIAD {S_['TRIAD']['mean']:.3f}±{S_['TRIAD']['sd']:.3f} static {S_['static']['mean']:.3f} "
          f"best {S_['best_competitor']:12s} {S_['best_value']['mean']:.3f} | vs best {S_['seeds_better_than_best']:2d}/10 "
          f"{S_['label_vs_best']:24s} | vs static {S_['seeds_better_than_static']:2d}/10 | J {S_['J']['mean']:.1f} "
          f"ARI {S_['ari_adapt']['mean']:.2f}")
