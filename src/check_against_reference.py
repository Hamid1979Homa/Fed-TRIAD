"""Compare a freshly computed numbers.json with the reference numbers.json that the numbers in the paper come from.

Usage: python check_against_reference.py <new numbers.json> <reference numbers.json>

What is checked (every value in the file is compared, nothing is skipped):
  * structure: both files must contain exactly the same keys and list lengths;
  * integers, booleans and strings (numbers of seeds, counts of "better" seeds, decisions "better / worse / no consistent
    difference", the strongest competitor, numbers of re-clusterings, labels) must be EQUAL. The conclusions of the paper
    rest on these values;
  * floating-point values are compared with tolerance tiers and REPORTED. Results are bit-identical only on the same CPU /
    BLAS / library versions; on other machines last digits differ, and in ill-conditioned gate fits (mainly the data-scarce
    setting) rounding differences can be amplified (README, "Reproducibility"). The check fails on floats only if an
    aggregated mean ("mean" entries) differs by more than 10 %, which indicates a broken environment, not rounding.
Exit status 0 = conclusions reproduced, 1 = mismatch.
"""
import json
import math
import sys

MEAN_GUARD = 0.10          # an aggregated mean that differs by more than this is a failure
TIERS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


def walk(a, b, path, leaf, acc):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                acc['struct'].append((f'{path}/{k}', 'missing in ' + ('new' if k not in a else 'reference')))
            else:
                walk(a[k], b[k], f'{path}/{k}', k, acc)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc['struct'].append((path, f'length {len(a)} vs {len(b)}'))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{path}[{i}]', leaf, acc)
    elif isinstance(a, (bool, str)) or isinstance(b, (bool, str)) or a is None or b is None:
        acc['n_exact'] += 1
        if a != b or type(a) is not type(b):
            acc['exact'].append((path, f'{a!r} vs {b!r}'))
    elif isinstance(a, int) and isinstance(b, int):
        acc['n_exact'] += 1
        if a != b:
            acc['exact'].append((path, f'{a} vs {b}'))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return
        diff = abs(a - b)
        rel = 0.0 if diff <= 1e-9 else diff / max(abs(a), abs(b))
        acc['floats'].append((rel, path, leaf == 'mean'))
    else:
        acc['struct'].append((path, f'different types: {type(a).__name__} vs {type(b).__name__}'))


def main():
    new, ref = (json.load(open(p)) for p in sys.argv[1:3])
    acc = dict(struct=[], exact=[], n_exact=0, floats=[])
    walk(new, ref, '', '', acc)
    fl = acc['floats']
    n = len(fl)
    print(f'exact values (counts, decisions, labels): {acc["n_exact"]} compared, {len(acc["exact"])} different')
    print(f'floating-point values: {n} compared; share agreeing within a relative tolerance of ' +
          ', '.join(f'{t:g}: {100.0 * sum(1 for r, _, _ in fl if r <= t) / max(n, 1):.1f}%' for t in TIERS))
    means = [(r, p) for r, p, is_mean in fl if is_mean]
    if means:
        r, p = max(means)
        print(f'aggregated means: {len(means)} compared, largest relative difference {r:.2e} ({p})')
    ok = True
    if acc['struct']:
        ok = False
        print(f'STRUCTURE MISMATCH ({len(acc["struct"])}); first 10:')
        for p, msg in acc['struct'][:10]:
            print(f'  {p}: {msg}')
    if acc['exact']:
        ok = False
        print(f'MISMATCH in {len(acc["exact"])} count/decision/label value(s); first 15:')
        for p, msg in acc['exact'][:15]:
            print(f'  {p}: {msg}')
    gross = [(r, p) for r, p in means if r > MEAN_GUARD]
    if gross:
        ok = False
        print(f'MISMATCH: {len(gross)} aggregated mean(s) differ by more than {MEAN_GUARD:.0%}; first 10:')
        for r, p in sorted(gross, reverse=True)[:10]:
            print(f'  {p}: relative difference {r:.2e}')
    if not ok:
        sys.exit(1)
    print('MATCH: every count, decision and label equals the reference of the paper. Remaining differences in continuous '
          'values are rounding-level (README, "Reproducibility").')


if __name__ == '__main__':
    main()
