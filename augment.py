"""
augment.py — Augmented-view voting for solver.py, inspired by the
Product-of-Experts approach (Franzen, Disselhoff, Hartmann, ICML 2025):
instead of trusting one derivation, re-derive the answer from several
different "perspectives" of the same task and keep whichever candidate
answer is consistent across the most perspectives.

Their version scores LLM-generated candidates by re-evaluating likelihood
under rotated/recolored views. We have no LLM, so we adapt the idea to
symbolic search: solve the SAME task 8 times, once per dihedral transform
(rotate/flip/transpose applied to every train pair AND the test input),
map each run's answer back to the original orientation, then vote.

If the true transformation rule is genuinely rotation/flip-invariant in
structure (most ARC geometric/color rules are), the 8 views should mostly
agree -- disagreement is a signal the plain single-view answer might be
wrong, and agreement across views is stronger evidence than either search
result alone.
"""

from collections import Counter

import dsl
from dsl import to_grid
import solver


# Each dihedral transform paired with its own inverse.
_DIHEDRAL = [
    ("identity", dsl.identity, dsl.identity),
    ("rotate90", dsl.rotate90, dsl.rotate270),
    ("rotate180", dsl.rotate180, dsl.rotate180),
    ("rotate270", dsl.rotate270, dsl.rotate90),
    ("flip_h", dsl.flip_h, dsl.flip_h),
    ("flip_v", dsl.flip_v, dsl.flip_v),
    ("transpose", dsl.transpose, dsl.transpose),
    ("anti_transpose", dsl.anti_transpose, dsl.anti_transpose),
]


def _transform_task(train, test_inputs, fwd):
    t_train = [(fwd(i), fwd(o)) for i, o in train]
    t_test = [fwd(t) for t in test_inputs]
    return t_train, t_test


def solve_with_voting(train, test_inputs, max_depth=2, min_votes=1):
    """Run search under all 8 dihedral views of the task, map every
    resulting candidate back to the original orientation, and rank by
    vote count. Returns (results, debug) where results mirrors
    solver.solve_task's shape: a list of up to-2 candidates per test input.
    """
    n_tests = len(test_inputs)
    # votes[i] : Counter mapping a candidate grid -> number of views that
    # produced it (only for grids that fit under that view's search).
    votes = [Counter() for _ in range(n_tests)]
    per_view_found = {}

    for name, fwd, inv in _DIHEDRAL:
        try:
            t_train, t_test = _transform_task(train, test_inputs, fwd)
            programs = solver.search(t_train, max_depth=max_depth)
        except Exception:
            programs = []
        per_view_found[name] = len(programs)
        if not programs:
            continue
        for i, t_grid in enumerate(t_test):
            seen_this_view = set()
            for program in programs:
                try:
                    t_out = solver._apply_program(program, t_grid)
                    out = inv(t_out)
                except Exception:
                    continue
                # count each view once per distinct candidate it proposes,
                # so a view with many equivalent programs doesn't dominate
                if out not in seen_this_view:
                    seen_this_view.add(out)
                    votes[i][out] += 1

    results = []
    for i in range(n_tests):
        ranked = [g for g, c in votes[i].most_common() if c >= min_votes]
        if not ranked:
            # nothing survived voting -> fall back to plain single-view search
            plain_preds, _ = solver.solve_task(train, [test_inputs[i]], max_depth=max_depth)
            ranked = plain_preds[0]
        results.append(ranked[:2])

    debug = {
        "views_with_solutions": sum(1 for v in per_view_found.values() if v > 0),
        "total_views": len(_DIHEDRAL),
        "votes_per_test": [dict(v) for v in votes],
    }
    return results, debug


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python augment.py path/to/task.json")
        sys.exit(0)
    train, test, test_out = solver.load_task(sys.argv[1])
    preds, debug = solve_with_voting(train, test)
    print(f"views with a solution: {debug['views_with_solutions']}/{debug['total_views']}")
    for i, cands in enumerate(preds):
        print(f"test[{i}] candidates (by vote):")
        for c in cands:
            print(dsl.to_list(c))
    if test_out:
        correct = sum(1 for gt, cands in zip(test_out, preds) if gt in cands)
        print(f"score: {correct}/{len(test_out)}")
