"""Compare the stored run trajectories against a fresh sweep."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_files import write_json
from run_sweep import SEEDS, SETTINGS, setting_config


def same_array(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare(actual, expected, path, state):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if actual.keys() != expected.keys():
            state["differences"].append(path)
            return
        for key in expected:
            compare(actual[key], expected[key], f"{path}/{key}", state)
    elif isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        state["arrays"] += 1
        if same_array(actual, expected):
            state["identical_arrays"] += 1
        else:
            state["differences"].append(path)
    elif type(actual) is not type(expected) or actual != expected:
        state["differences"].append(path)


def compare_sweep(actual_root, reference_root):
    state = dict(runs=0, arrays=0, identical_arrays=0, differences=[])
    for name in SETTINGS:
        settings = json.loads((actual_root / f"{name}.cfg.json").read_text(encoding="utf-8"))
        archive = json.loads((reference_root / f"{name}.cfg.json").read_text(encoding="utf-8"))
        # Two default options are now explicit in the manifest. Both were implicit in the original study.
        archive["CFG"].setdefault("eta", 1.0)
        archive["CFG"].setdefault("match_cluster_batches", False)
        if settings != setting_config(name) or settings != archive:
            state["differences"].append(f"{name}/configuration")
        fresh = np.load(actual_root / f"{name}.npy", allow_pickle=True).item()
        archived = np.load(reference_root / f"{name}.npy", allow_pickle=True).item()
        if sorted(fresh) != SEEDS or sorted(archived) != SEEDS:
            state["differences"].append(f"{name}/seeds")
            continue
        for seed in SEEDS:
            compare(fresh[seed], archived[seed], f"{name}/{seed}", state)
            state["runs"] += 1
    differences = state.pop("differences")
    return dict(state, identical=not differences, total_differences=len(differences),
                examples=differences[:20])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    report = compare_sweep(args.actual, args.reference)
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return int(args.require_exact and not report["identical"])


if __name__ == "__main__":
    raise SystemExit(main())
