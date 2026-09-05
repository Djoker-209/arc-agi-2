"""
Synthetic sanity checks. Real ARC-AGI-2 task JSON isn't reachable from this
sandbox (kaggle.com / arcprize.org aren't in the allowed network domains),
so these hand-built tasks stand in to prove the DSL + search actually work
end-to-end before you drop in real task files.
"""

import dsl
from dsl import to_grid
from solver import search, solve_task, _apply_program


def make_task(pairs, test_in, test_out=None):
    train = [(to_grid(i), to_grid(o)) for i, o in pairs]
    test = [to_grid(test_in)]
    gt = [to_grid(test_out)] if test_out else None
    return train, test, gt


def check(name, train, test, gt):
    preds, programs = solve_task(train, test)
    ok = gt is not None and gt[0] in preds[0]
    print(f"[{name}] programs_found={len(programs)} "
          f"candidates={[dsl.to_list(c) for c in preds[0]]} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


results = []

# 1) rotate90
train, test, gt = make_task(
    pairs=[
        ([[1, 0], [0, 0]], [[0, 1], [0, 0]]),
        ([[2, 0], [0, 0]], [[0, 2], [0, 0]]),
    ],
    test_in=[[3, 0], [0, 0]],
    test_out=[[0, 3], [0, 0]],
)
results.append(check("rotate90", train, test, gt))

# 2) flip_h
train, test, gt = make_task(
    pairs=[
        ([[1, 2], [3, 4]], [[2, 1], [4, 3]]),
        ([[5, 6], [7, 8]], [[6, 5], [8, 7]]),
    ],
    test_in=[[9, 0], [1, 2]],
    test_out=[[0, 9], [2, 1]],
)
results.append(check("flip_h", train, test, gt))

# 3) recolor mapping (1->2, 2->3)
train, test, gt = make_task(
    pairs=[
        ([[1, 1], [2, 0]], [[2, 2], [3, 0]]),
        ([[0, 2], [1, 1]], [[0, 3], [2, 2]]),
    ],
    test_in=[[1, 2], [2, 1]],
    test_out=[[2, 3], [3, 2]],
)
results.append(check("recolor", train, test, gt))

# 4) 2x upscale
train, test, gt = make_task(
    pairs=[
        ([[1, 2]], [[1, 1, 2, 2], [1, 1, 2, 2]]),
    ],
    test_in=[[3, 4]],
    test_out=[[3, 3, 4, 4], [3, 3, 4, 4]],
)
results.append(check("scale2x", train, test, gt))

# 5) composed depth-2: rotate180 then flip_h == flip_v
train, test, gt = make_task(
    pairs=[
        ([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
        ([[5, 6], [7, 8]], [[7, 8], [5, 6]]),
    ],
    test_in=[[9, 0], [1, 2]],
    test_out=[[1, 2], [9, 0]],
)
results.append(check("depth2_flip_v_equiv", train, test, gt))

# 6) crop_to_content
train, test, gt = make_task(
    pairs=[
        ([[0, 0, 0], [0, 5, 0], [0, 0, 0]], [[5]]),
        ([[0, 0], [0, 7]], [[7]]),
    ],
    test_in=[[0, 0, 0], [0, 0, 9]],
    test_out=[[9]],
)
results.append(check("crop_to_content", train, test, gt))

print(f"\n{sum(results)}/{len(results)} synthetic tasks passed")

