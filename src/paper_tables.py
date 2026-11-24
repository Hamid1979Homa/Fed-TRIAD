"""Build the numerical cells of Tables 3, 4 and 6 from analyzed simulations."""

import csv
import json
from pathlib import Path


BASE = "drift_m12"
MAIN = (BASE, "drift_m30", "scarce_m12")
METHODS = (
    "Local", "FedAvg", "CFL-static", "CFL-adaptive", "PG-const", "PG-gate",
    "PC-gate", "Ditto", "TRIAD-uniform", "TRIAD-const", "TRIAD-static",
    "TRIAD", "TRIAD-oracle",
)
GATES = ("qp", "best_single", "learned", "diag", "uniform", "resid_corr", "thm5_ratio")
HEADERS = {
    "table3": ["Method", "Final risk m=12", "Regret m=12", "Regret slope",
               "Final risk m=30", "Final risk scarce"],
    "table4": ["Gate or statistic", "m=12", "m=30", "Scarce"],
    "table6": ["Setting", "Fed-TRIAD", "Static partition", "Strongest competitor",
               "Lower than strongest", "Lower than static", "J", "Final ARI"],
}


def mean_sd(stat, digits=3):
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def paper_tables(numbers):
    table3 = {}
    for method in METHODS:
        base = numbers[BASE][method]
        table3[method] = [
            mean_sd(base["final"]), mean_sd(base["regret"], 2), mean_sd(base["slope_last50"]),
            mean_sd(numbers["drift_m30"][method]["final"]),
            mean_sd(numbers["scarce_m12"][method]["final"]),
        ]
    table4 = {gate: [mean_sd(numbers[setting]["_diag"][gate]) for setting in MAIN] for gate in GATES}
    table6 = {}
    for setting, row in numbers["_sens"].items():
        n = row["TRIAD"]["n"]
        method = "P/C gate" if row["best_competitor"] == "PC-gate" else row["best_competitor"]
        table6[setting] = [
            mean_sd(row["TRIAD"]), mean_sd(row["static"]),
            f"{method}: {mean_sd(row['best_value'])}",
            f"{row['seeds_better_than_best']}/{n}, {row['label_vs_best']}",
            f"{row['seeds_better_than_static']}/{n}",
            f"{row['J']['mean']:.1f}", f"{row['ari_adapt']['mean']:.2f}",
        ]
    return {"table3": table3, "table4": table4, "table6": table6}


def write_tables(numbers, directory):
    directory = Path(directory)
    tables = paper_tables(numbers)
    for name, rows in tables.items():
        with (directory / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADERS[name])
            writer.writerows([row, *cells] for row, cells in rows.items())
    (directory / "paper_tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return tables
