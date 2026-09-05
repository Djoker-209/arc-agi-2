import dsl
from dsl import to_grid
from augment import solve_with_voting


def make_task(pairs, test_in, test_out=None):
    train = [(to_grid(i), to_grid(o)) for i, o in pairs]
    test = [to_grid(test_in)]
    gt = [to_grid(test_out)] if test_out else None
    return train, test, gt


results = []

# 1) plain rotate90 -- voting should still get this right
train, test, gt = make_task(
    pairs=[
        ([[1, 0], [0, 0]], [[0, 1], [0, 0]]),
        ([[2, 0], [0, 0]], [[0, 2], [0, 0]]),
    ],
    test_in=[[3, 0], [0, 0]],
    test_out=[[0, 3], [0, 0]],
)
preds, debug = solve_with_voting(train, test)
ok = gt[0] in preds[0]
print(f"[rotate90 via voting] views_with_solutions={debug['views_with_solutions']}/8 "
      f"candidates={[dsl.to_list(c) for c in preds[0]]} {'PASS' if ok else 'FAIL'}")
results.append(ok)

# 2) ambiguous case: a 1x1 -> 1x1 identity-shaped task where several ops
# (identity, and recolor-with-no-change) trivially agree -- checks voting
# doesn't break on a degenerate/trivial task.
train, test, gt = make_task(
    pairs=[([[5]], [[5]]), ([[3]], [[3]])],
    test_in=[[7]],
    test_out=[[7]],
)
preds, debug = solve_with_voting(train, test)
ok = gt[0] in preds[0]
print(f"[trivial identity] views_with_solutions={debug['views_with_solutions']}/8 "
      f"candidates={[dsl.to_list(c) for c in preds[0]]} {'PASS' if ok else 'FAIL'}")
results.append(ok)

# 3) flip_v task -- confirm inverse-mapping correctness end to end
train, test, gt = make_task(
    pairs=[
        ([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
        ([[5, 6], [7, 8]], [[7, 8], [5, 6]]),
    ],
    test_in=[[9, 0], [1, 2]],
    test_out=[[1, 2], [9, 0]],
)
preds, debug = solve_with_voting(train, test)
ok = gt[0] in preds[0]
print(f"[flip_v via voting] views_with_solutions={debug['views_with_solutions']}/8 "
      f"candidates={[dsl.to_list(c) for c in preds[0]]} {'PASS' if ok else 'FAIL'}")
results.append(ok)

print(f"\n{sum(results)}/{len(results)} voting tests passed")
