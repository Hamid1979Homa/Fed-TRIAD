"""Compare recomputed statistics with the archived run and the printed Word cells."""

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path

from paper_tables import paper_tables


def differences(actual, reference, path="", *, floats):
    if type(actual) is not type(reference):
        yield path, "structure", f"{type(actual).__name__} != {type(reference).__name__}"
    elif isinstance(reference, dict):
        for key in sorted(reference.keys() | actual.keys()):
            field = f"{path}/{key}"
            if key in reference and key in actual:
                yield from differences(actual[key], reference[key], field, floats=floats)
            else:
                yield field, "structure", "missing or extra key"
    elif isinstance(reference, list):
        if len(actual) != len(reference):
            yield path, "structure", f"length {len(actual)} != {len(reference)}"
        for i, (a, b) in enumerate(zip(actual, reference)):
            yield from differences(a, b, f"{path}[{i}]", floats=floats)
    elif type(reference) is float:
        finite = math.isfinite(actual) and math.isfinite(reference)
        if not finite:
            if math.isnan(actual) and math.isnan(reference) and "/_first_post_shift_recluster/" in path:
                return
            yield path, "nonfinite", f"{actual!r} vs {reference!r}"
            return
        exact = struct.pack("!d", actual) == struct.pack("!d", reference)
        floats["count"] += 1
        floats["identical"] += exact
        if not exact:
            absdiff = abs(actual - reference)
            rel = absdiff / max(abs(actual), abs(reference), 1e-300)
            if absdiff > floats["max_abs"]:
                floats["max_abs"], floats["max_abs_path"] = absdiff, path
            if rel > floats["max_rel"]:
                floats["max_rel"], floats["max_rel_path"] = rel, path
            yield path, "float", f"{actual!r} vs {reference!r}"
    elif actual != reference:
        yield path, "exact", f"{actual!r} vs {reference!r}"


def float_summary():
    return dict(count=0, identical=0, max_abs=0.0, max_rel=0.0,
                max_abs_path=None, max_rel_path=None)


def headline_checks(numbers):
    sensitivity = numbers["_sens"]
    return {
        "adaptive_better_than_static_settings": sum(row["label_vs_static"] == "better"
                                                     for row in sensitivity.values()),
        "pc_better_than_three_path_settings": sum(row["label_vs_PC"] == "worse"
                                                 for row in sensitivity.values()),
        "recluster_jumps": {
            setting: [len(values), sum(x > 0.0 for x in values)]
            for setting in ("drift_m12", "drift_m30", "scarce_m12")
            if (values := numbers[setting]["_recluster_jumps"]) is not None
        },
    }


def audit(actual, reference, printed, criterion):
    floats = float_summary()
    delta = list(differences(actual, reference, floats=floats))
    counts = dict(Counter(kind for _, kind, _ in delta))
    problems = [item for item in delta if criterion == "exact" or item[1] != "float"]
    if counts.get("structure") or counts.get("nonfinite"):
        return dict(criterion=criterion, passed=False, table_cells=0, floats=floats,
                    differences=counts, examples=problems[:12])
    table_diffs = list(differences(paper_tables(actual), printed, floats=float_summary()))
    headlines = headline_checks(actual)
    headline_diffs = list(differences(headlines, headline_checks(reference), floats=float_summary()))
    table_cells = sum(len(row) for table in printed.values() for row in table.values())
    return dict(criterion=criterion, passed=not (problems or table_diffs or headline_diffs),
                table_cells=table_cells, mismatching_cells=len(table_diffs),
                mismatching_headlines=len(headline_diffs), headlines=headlines, floats=floats,
                differences=counts, examples=(table_diffs + headline_diffs + problems)[:12])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--printed", type=Path, default=Path(__file__).resolve().parents[1] / "reference/paper_tables.json")
    parser.add_argument("--criterion", choices=("exact", "tables"), default="exact")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    actual, reference, printed = (json.loads(p.read_text(encoding="utf-8"))
                                  for p in (args.actual, args.reference, args.printed))
    report = audit(actual, reference, printed, args.criterion)
    body = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report:
        args.report.write_text(body + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
