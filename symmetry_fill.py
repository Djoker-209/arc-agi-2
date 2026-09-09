"""
symmetry_fill.py — Solves the common ARC-AGI-2 task family where a large,
mostly-symmetric grid has a solid-color rectangular "hole" punched in it,
and the expected output is just the reconstructed content of that hole
(not the whole grid).

Key finding (verified against a real ARC-AGI-2 public eval task): the
mirror symmetry axis is frequently NOT at the exact grid center. A naive
flip_h/flip_v/rotate180 overlay (which assumes center symmetry) fails
silently -- it produces a plausible-looking but wrong answer. The fix is
to search for the actual best-fit axis position (row axis, column axis,
or both) by minimizing mismatches over all non-occluded cells, then
reconstruct using that axis.

Approach:
  1. Find the occluder color: the color whose cells form a single solid
     filled rectangle (bbox area == cell count).
  2. Search over all possible row-mirror and column-mirror axis
     positions (including half-integer / off-center axes) for the one
     with zero (or minimal) mismatches among non-occluded cells.
  3. Reconstruct occluded cells using the row axis, column axis, and
     their combination (point symmetry through the axis intersection),
     whichever gives a non-occluded value.
  4. Crop the reconstructed grid to the occluder's bounding box.
"""

from collections import Counter

import dsl
from dsl import dims, to_grid


def find_occluder_color(grid, exclude=()):
    """Return (color, bbox, area) for the color whose cells form exactly
    one solid filled rectangle, or None if no such color exists."""
    h, w = dims(grid)
    color_cells = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v in exclude:
                continue
            color_cells.setdefault(v, []).append((r, c))

    candidates = []
    for color, cells in color_cells.items():
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area == len(cells) and area < h * w:
            candidates.append((color, (r0, c0, r1, c1), area))

    if not candidates:
        return None
    most_common = Counter(v for row in grid for v in row).most_common(1)[0][0]
    candidates = [c for c in candidates if c[0] != most_common] or candidates
    candidates.sort(key=lambda x: -x[2])
    return candidates[0]


def _best_axis(grid, occluder, axis_kind, min_overlap_frac=0.3):
    """Search all mirror-axis positions (in half-integer units, so an
    axis between two cells is representable) for the one with the fewest
    mismatches among non-occluded cells. axis_kind: 'row' or 'col'.
    Returns (axis_x2, mismatches, overlap_checked) or None if nothing
    clears the minimum-overlap bar (too little evidence to trust it)."""
    h, w = dims(grid)
    size = h if axis_kind == "row" else w
    other_size = w if axis_kind == "row" else h
    best = None

    for axis2 in range(0, 2 * size - 1):
        mism = 0
        checked = 0
        for i in range(size):
            mi = axis2 - i
            if mi < 0 or mi >= size or mi == i:
                continue
            for j in range(other_size):
                a = grid[i][j] if axis_kind == "row" else grid[j][i]
                b = grid[mi][j] if axis_kind == "row" else grid[j][mi]
                if a == occluder or b == occluder:
                    continue
                checked += 1
                if a != b:
                    mism += 1
        min_overlap = min_overlap_frac * size * other_size
        if checked >= min_overlap and (best is None or mism < best[1]):
            best = (axis2, mism, checked)
    return best


def reconstruct(grid, occluder):
    """Find best row/col mirror axes and use them (plus their
    combination, i.e. point symmetry) to fill occluded cells."""
    h, w = dims(grid)
    row_best = _best_axis(grid, occluder, "row")
    col_best = _best_axis(grid, occluder, "col")

    row_axis2 = row_best[0] if row_best and row_best[1] == 0 else None
    col_axis2 = col_best[0] if col_best and col_best[1] == 0 else None

    if row_axis2 is None and col_axis2 is None:
        return None  # no exact symmetry found -- don't guess

    out = [list(row) for row in grid]
    for r in range(h):
        for c in range(w):
            if out[r][c] != occluder:
                continue
            mr = row_axis2 - r if row_axis2 is not None else None
            mc = col_axis2 - c if col_axis2 is not None else None
            candidates = []
            if mr is not None and 0 <= mr < h and grid[mr][c] != occluder:
                candidates.append(grid[mr][c])
            if mc is not None and 0 <= mc < w and grid[r][mc] != occluder:
                candidates.append(grid[r][mc])
            if (mr is not None and mc is not None and 0 <= mr < h and 0 <= mc < w
                    and grid[mr][mc] != occluder):
                candidates.append(grid[mr][mc])
            if candidates:
                out[r][c] = candidates[0]
    return to_grid(out)


def solve(train, test_inputs):
    """Returns a list of candidate outputs per test input (empty list per
    test if the strategy doesn't confidently apply to that case),
    mirroring solver.py's interface. Requires an EXACT fit on every train
    pair before trusting any test prediction."""
    for inp, out in train:
        found = find_occluder_color(inp)
        if found is None:
            return []
        color, bbox, _ = found
        r0, c0, r1, c1 = bbox
        recon = reconstruct(inp, color)
        if recon is None:
            return []
        crop = dsl.crop(recon, r0, c0, r1, c1)
        if crop != out:
            return []

    results = []
    for grid in test_inputs:
        found = find_occluder_color(grid)
        if found is None:
            results.append([])
            continue
        color, bbox, _ = found
        r0, c0, r1, c1 = bbox
        recon = reconstruct(grid, color)
        if recon is None:
            results.append([])
            continue
        crop = dsl.crop(recon, r0, c0, r1, c1)
        # If any occluder-colored cells survive reconstruction, the axis
        # we found doesn't actually cover this occlusion's position (e.g.
        # it sits in a corner region with no in-bounds mirror partner).
        # Refuse to guess rather than silently leak the occluder color
        # into the answer -- an honest "no answer" beats a confidently
        # wrong one under exact-match scoring.
        if any(v == color for row in crop for v in row):
            results.append([])
            continue
        results.append([crop])
    return results


if __name__ == "__main__":
    import sys
    import solver
    train, test, test_out = solver.load_task(sys.argv[1])
    preds = solve(train, test)
    for i, cands in enumerate(preds):
        print(f"test[{i}] candidates:")
        for c in cands:
            print(dsl.to_list(c))
    if test_out:
        correct = sum(1 for gt, cands in zip(test_out, preds) if gt in cands)
        print(f"score: {correct}/{len(test_out)}")
