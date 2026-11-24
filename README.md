# Fed-TRIAD

Code for the numerical study in Section 6 of the Fed-TRIAD manuscript, version
`FedTRIAD_clean_EndNote_v3`. The study simulates personalized federated learning
with input-dependent personal, cluster and global predictions and adaptive
reclustering. The data generator uses synthetic multi-output regression.

## Reproduce the manuscript

On Linux x86-64 with x86-64-v4 instructions, install the wheel versions used for
the exact-reference profile with Python 3.13.5. From the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements-lock.txt
.venv/bin/python reproduce.py --out reproduced --jobs 2
```

The runner trains 23 configurations with 10 seeds each. It writes the new runs
and tables to the chosen directory. It checks all printed cells in Tables 3, 4
and 6, the headline counts, the full-precision statistics, and the per-round
arrays against the archived reference. The procedure takes tens of minutes or
longer depending on the CPU and the number of workers. Use the same command with
`--resume` to continue an interrupted run with matching settings and environment.

On other CPU profiles or operating systems, use `--mode native` to try the study
with the same package versions and the system's numerical kernels. It requires
the same printed cells and headline counts and records differences in the raw
arrays and full-precision statistics. A failed verification returns a nonzero
exit status and names the differences in the report. NumPy and the libraries it
uses can dispatch different floating-point routines on different CPU and OS
combinations.

<details>
<summary>Windows PowerShell</summary>

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --require-hashes -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduce.py --mode native --out reproduced --jobs 2
```

</details>

## Change a setting

This command runs a new simulation with three seeds and a higher reclustering
hysteresis threshold. Each seed goes through the same training and evaluation
code as the manuscript study.

```bash
.venv/bin/python src/fedtriad_sim.py --m 12 --seeds 1 2 3 --out exploration/hysteresis.npy --set tau_imp=0.10
```

The command reports the late-round risks, the number of accepted reclusterings,
and final partition agreement. Its companion `.cfg.json` records the resolved
configuration. The switches `--scenario scarce` and `--set` support sample counts,
noise, concept separation (`kappa`), the number of clients changing concept
(`s`), the gate and path learning rates, and other settings in `src/fedtriad_sim.py`.
Distinct experiments require distinct output paths. Add
`--set match_cluster_batches=true` to match minibatch indices across the three
cluster regimes for a paired control.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Files and results

| File or directory | Contents |
|---|---|
| `PROTOCOL.md` | Algorithms, random stream order, metric definitions, baseline sources and adaptations |
| `src/fedtriad_sim.py` | Data generator, training, gating, clustering and diagnostics |
| `src/run_sweep.py` | Complete manuscript grid and verified resume |
| `src/analyze.py` | Aggregation of the generated runs |
| `reproduced/outputs/table{3,4,6}.csv` | Regenerated manuscript table cells at their printed precision |
| `reproduced/outputs/numbers.json` | Full-precision statistics and per-seed values |
| `reproduced/outputs/fig*.png` | Regenerated Figures 2–6. Plot typography depends on environment fonts. |
| `reproduced/results/study.json` | Resolved configurations and numerical environment |
| `reproduced/{numbers,raw}_verification.json` | Comparison reports |
| `results/` and `reference/` | Archived trajectories, source statistics and a transcription of the printed tables |

The array archives contain Python objects. Load these `.npy` files only from a
trusted checkout or your own simulation.

MIT licence. See [LICENSE](LICENSE).
