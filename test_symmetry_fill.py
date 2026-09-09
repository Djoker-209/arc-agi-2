import dsl
import symmetry_fill


def test_offcenter_axis():
    """8x8 grid, mirror axis between columns 5 and 6 (NOT the center,
    which would be between 3 and 4) -- checks the axis search actually
    finds an off-center axis instead of assuming the grid midpoint."""
    # Build a grid that is symmetric about column-axis x2=9 (axis at 4.5,
    # deliberately NOT the true grid center at 3.5) by construction, then
    # punch an occluder hole whose mirror partners stay in-bounds.
    import random
    random.seed(0)
    w, h = 8, 5
    base = [[random.randint(1, 6) for _ in range(w)] for _ in range(h)]
    grid = [row[:] for row in base]
    for r in range(h):
        for c in range(w):
            mc = 9 - c
            if 0 <= mc < w:
                grid[r][mc] = grid[r][c]

    true_grid = dsl.to_grid(grid)
    occluded = [list(row) for row in grid]
    # punch a solid rectangle hole with color 8 (columns 1-2 mirror to
    # columns 8,7 -> only column 1->8 is out of range; use col 2-3 instead
    # whose mirrors are 7,6, both in-bounds)
    for r in range(1, 3):
        for c in range(2, 4):
            occluded[r][c] = 8
    occluded = dsl.to_grid(occluded)

    expected_patch = dsl.crop(true_grid, 1, 2, 2, 3)

    train = [(occluded, expected_patch)]
    test = [occluded]
    preds = symmetry_fill.solve(train, test)
    ok = bool(preds) and bool(preds[0]) and expected_patch in preds[0]
    print(f"[offcenter_axis] candidates={[dsl.to_list(c) for c in preds[0]] if preds and preds[0] else preds} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_no_symmetry_returns_empty():
    """A grid with no real symmetry at all should make solve() bail out
    (return []) rather than guess wrong."""
    grid = dsl.to_grid([
        [1, 2, 3, 8, 8],
        [4, 5, 6, 8, 8],
        [7, 1, 2, 3, 4],
    ])
    fake_out = dsl.to_grid([[9, 9], [9, 9]])
    train = [(grid, fake_out)]  # deliberately wrong/unrelated output
    preds = symmetry_fill.solve(train, [grid])
    ok = preds == []
    print(f"[no_symmetry_returns_empty] result={preds} {'PASS' if ok else 'FAIL'}")
    return ok


results = [test_offcenter_axis(), test_no_symmetry_returns_empty()]
print(f"\n{sum(results)}/{len(results)} symmetry_fill tests passed")
