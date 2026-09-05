"""
solver.py — Baseline 0 for ARC-AGI-2: brute-force DSL composition search.

Strategy:
  1. Try each single unary op from dsl.UNARY_OPS.
  2. Try each pair of unary ops composed (depth 2).
  3. Try parametrized ops with parameters *inferred from the train pairs*
     (scale/downscale factor from size ratio, recolor mapping from color
     correspondence, tile reps from size ratio).
  4. A candidate program is accepted only if it reproduces EVERY train
     pair's output exactly.
  5. Return up to 2 candidate outputs for each test input (competition
     scoring allows 2 guesses per test case).

This is intentionally simple and fast (no ML) so it runs in milliseconds
per task -- it exists to (a) solve the subset of tasks that really are
single/double geometric or color transforms, and (b) act as a fallback
signal when a heavier model-based solver (test-time adaptation, etc.)
doesn't produce a confident answer.
"""

import json
from itertools import product

import dsl
from dsl import to_grid, dims


def load_task(path):
    with open(path) as f:
        raw = json.load(f)
    train = [(to_grid(p["input"]), to_grid(p["output"])) for p in raw["train"]]
    test = [to_grid(p["input"]) for p in raw["test"]]
    test_out = None
    if "output" in raw["test"][0]:
        test_out = [to_grid(p["output"]) for p in raw["test"]]
    return train, test, test_out


# ---------------------------------------------------------------- candidate programs

def _try_program(program, train):
    """program: list of (fn, kwargs). Returns True if it fits every train pair."""
    for inp, out in train:
        g = inp
        try:
            for fn, kwargs in program:
                g = fn(g, **kwargs) if kwargs else fn(g)
        except Exception:
            return False
        if g != out:
            return False
    return True


def _apply_program(program, grid):
    g = grid
    for fn, kwargs in program:
        g = fn(g, **kwargs) if kwargs else fn(g)
    return g


def _infer_recolor_mapping(train):
    """If every train pair shares the same input->output color mapping
    (same shape, cellwise recolor), infer and return that mapping dict."""
    mapping = {}
    for inp, out in train:
        if dims(inp) != dims(out):
            return None
        for r_in, r_out in zip(inp, out):
            for a, b in zip(r_in, r_out):
                if a in mapping and mapping[a] != b:
                    return None
                mapping[a] = b
    return mapping


def _infer_scale_factors(train):
    """If output is a consistent integer multiple of input size, return (fr, fc)."""
    ratios = set()
    for inp, out in train:
        hi, wi = dims(inp)
        ho, wo = dims(out)
        if hi == 0 or wi == 0 or ho % hi or wo % wi:
            return None
        ratios.add((ho // hi, wo // wi))
    return ratios.pop() if len(ratios) == 1 else None


def _infer_downscale_factors(train):
    ratios = set()
    for inp, out in train:
        hi, wi = dims(inp)
        ho, wo = dims(out)
        if ho == 0 or wo == 0 or hi % ho or wi % wo:
            return None
        ratios.add((hi // ho, wi // wo))
    return ratios.pop() if len(ratios) == 1 else None


def search(train, max_depth=2, time_budget_ops=200_000):
    """Return a list of programs (each a list of (fn, kwargs) steps) that
    exactly reproduce every train pair, ordered simplest-first."""
    found = []
    ops_tried = 0

    # depth 1: single unary op
    unary_items = list(dsl.UNARY_OPS.items())
    for name, fn in unary_items:
        ops_tried += 1
        if _try_program([(fn, None)], train):
            found.append([(fn, None)])

    # depth 1: parametrized ops with inferred params
    mapping = _infer_recolor_mapping(train)
    if mapping and _try_program([(dsl.recolor, {"mapping": mapping})], train):
        found.append([(dsl.recolor, {"mapping": mapping})])

    scale_f = _infer_scale_factors(train)
    if scale_f:
        prog = [(dsl.scale, {"factor_r": scale_f[0], "factor_c": scale_f[1]})]
        if _try_program(prog, train):
            found.append(prog)
        prog2 = [(dsl.tile, {"reps_r": scale_f[0], "reps_c": scale_f[1]})]
        if _try_program(prog2, train):
            found.append(prog2)

    down_f = _infer_downscale_factors(train)
    if down_f:
        prog = [(dsl.downscale, {"factor_r": down_f[0], "factor_c": down_f[1]})]
        if _try_program(prog, train):
            found.append(prog)

    if found or max_depth < 2:
        return found

    # depth 2: compose pairs of unary ops
    for (n1, f1), (n2, f2) in product(unary_items, repeat=2):
        if ops_tried > time_budget_ops:
            break
        ops_tried += 1
        program = [(f1, None), (f2, None)]
        if _try_program(program, train):
            found.append(program)

    return found


def solve_task(train, test_inputs, max_depth=2):
    """Return, for each test input, a list of up to 2 candidate output grids."""
    programs = search(train, max_depth=max_depth)
    results = []
    for grid in test_inputs:
        candidates = []
        for program in programs:
            try:
                out = _apply_program(program, grid)
            except Exception:
                continue
            if out not in candidates:
                candidates.append(out)
            if len(candidates) == 2:
                break
        if not candidates:
            # Fallback guesses: identity, then most-common-recolor-of-input.
            candidates = [grid]
            mc = dsl.most_common_color(grid, exclude_zero=True)
            alt = dsl.recolor(grid, {0: mc}) if mc else grid
            if alt != grid:
                candidates.append(alt)
        results.append(candidates[:2])
    return results, programs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python solver.py path/to/task.json")
        sys.exit(0)
    train, test, test_out = load_task(sys.argv[1])
    preds, programs = solve_task(train, test)
    for i, cands in enumerate(preds):
        print(f"test[{i}] candidates:")
        for c in cands:
            print(dsl.to_list(c))
    if test_out:
        correct = sum(
            1 for gt, cands in zip(test_out, preds) if gt in cands
        )
        print(f"score: {correct}/{len(test_out)}")

