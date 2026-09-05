"""
dsl.py — Primitive grid operations for ARC-AGI-2 program synthesis.

A "grid" is a tuple of tuples of ints 0-9 (immutable, hashable -> usable in
search state / memoization). All primitives take a grid (+ params) and
return a grid. Kept pure and side-effect free so they compose safely.
"""

from collections import Counter, deque
from itertools import product

Grid = tuple  # tuple[tuple[int, ...], ...]


def to_grid(rows):
    """list[list[int]] -> immutable Grid."""
    return tuple(tuple(r) for r in rows)


def to_list(grid):
    """Grid -> list[list[int]] for JSON output."""
    return [list(r) for r in grid]


def dims(grid):
    return len(grid), (len(grid[0]) if grid else 0)


# ---------------------------------------------------------------- geometry

def identity(grid):
    return grid


def rotate90(grid):
    return tuple(zip(*grid[::-1]))


def rotate180(grid):
    return tuple(row[::-1] for row in grid[::-1])


def rotate270(grid):
    return tuple(zip(*grid))[::-1]


def flip_h(grid):
    """Mirror left-right."""
    return tuple(row[::-1] for row in grid)


def flip_v(grid):
    """Mirror top-bottom."""
    return grid[::-1]


def transpose(grid):
    return tuple(zip(*grid))


def anti_transpose(grid):
    return flip_h(flip_v(transpose(grid)))


GEOMETRY_OPS = [identity, rotate90, rotate180, rotate270,
                flip_h, flip_v, transpose, anti_transpose]


# ---------------------------------------------------------------- color

def most_common_color(grid, exclude_zero=False):
    counts = Counter(v for row in grid for v in row)
    if exclude_zero:
        counts.pop(0, None)
    if not counts:
        return 0
    return counts.most_common(1)[0][0]


def recolor(grid, mapping):
    """mapping: dict old_color -> new_color."""
    return tuple(tuple(mapping.get(v, v) for v in row) for row in grid)


def swap_colors(grid, a, b):
    return recolor(grid, {a: b, b: a})


def replace_color(grid, old, new):
    return recolor(grid, {old: new})


def palette(grid):
    return sorted(set(v for row in grid for v in row))


# ---------------------------------------------------------------- crop / pad / tile

def bounding_box(grid, background=0):
    """Bounding box of all non-background cells. Returns (r0, c0, r1, c1) inclusive,
    or None if grid is all background."""
    cells = [(r, c) for r, row in enumerate(grid)
             for c, v in enumerate(row) if v != background]
    if not cells:
        return None
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def crop_to_content(grid, background=0):
    box = bounding_box(grid, background)
    if box is None:
        return grid
    r0, c0, r1, c1 = box
    return tuple(row[c0:c1 + 1] for row in grid[r0:r1 + 1])


def crop(grid, r0, c0, r1, c1):
    return tuple(row[c0:c1 + 1] for row in grid[r0:r1 + 1])


def pad(grid, top, bottom, left, right, fill=0):
    h, w = dims(grid)
    new_w = w + left + right
    out = []
    for _ in range(top):
        out.append(tuple([fill] * new_w))
    for row in grid:
        out.append(tuple([fill] * left) + row + tuple([fill] * right))
    for _ in range(bottom):
        out.append(tuple([fill] * new_w))
    return tuple(out)


def tile(grid, reps_r, reps_c):
    h, w = dims(grid)
    return tuple(
        tuple(grid[r % h][c % w] for c in range(w * reps_c))
        for r in range(h * reps_r)
    )


def scale(grid, factor_r, factor_c):
    """Nearest-neighbor upscale."""
    out = []
    for row in grid:
        new_row = tuple(v for v in row for _ in range(factor_c))
        out.extend([new_row] * factor_r)
    return tuple(out)


def downscale(grid, factor_r, factor_c):
    h, w = dims(grid)
    return tuple(
        tuple(grid[r * factor_r][c * factor_c] for c in range(w // factor_c))
        for r in range(h // factor_r)
    )


# ---------------------------------------------------------------- symmetry completion

def symmetrize(grid, axis="h", background=0):
    """Overlay grid with its mirror (h, v, or both) to fill background cells,
    assuming the true pattern is symmetric and only occluded by background."""
    h_flip, v_flip = flip_h(grid), flip_v(grid)
    h, w = dims(grid)
    out = [list(row) for row in grid]
    for r in range(h):
        for c in range(w):
            if out[r][c] != background:
                continue
            if axis in ("h", "both") and h_flip[r][c] != background:
                out[r][c] = h_flip[r][c]
            elif axis in ("v", "both") and v_flip[r][c] != background:
                out[r][c] = v_flip[r][c]
    return to_grid(out)


# ---------------------------------------------------------------- connected components / objects

def connected_components(grid, background=0, diagonal=False):
    """Return list of objects, each a dict: {cells: [(r,c,color)], bbox: (r0,c0,r1,c1)}.
    Cells of the same nonzero color connected via 4- (or 8-) adjacency."""
    h, w = dims(grid)
    seen = [[False] * w for _ in range(h)]
    if diagonal:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    objects = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or grid[r0][c0] == background:
                continue
            color = grid[r0][c0]
            q = deque([(r0, c0)])
            seen[r0][c0] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c, grid[r][c]))
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and not seen[nr][nc]
                            and grid[nr][nc] != background):
                        seen[nr][nc] = True
                        q.append((nr, nc))
            rs = [r for r, _, _ in cells]
            cs = [c for _, c, _ in cells]
            objects.append({
                "cells": cells,
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "color": color,
                "size": len(cells),
            })
    return objects


def largest_object(grid, background=0, diagonal=False):
    objs = connected_components(grid, background, diagonal)
    if not objs:
        return None
    return max(objs, key=lambda o: o["size"])


def object_to_grid(obj, background=0):
    r0, c0, r1, c1 = obj["bbox"]
    out = [[background] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
    for r, c, v in obj["cells"]:
        out[r - r0][c - c0] = v
    return to_grid(out)


# ---------------------------------------------------------------- flood fill

def flood_fill(grid, r, c, new_color):
    h, w = dims(grid)
    target = grid[r][c]
    if target == new_color:
        return grid
    out = [list(row) for row in grid]
    q = deque([(r, c)])
    while q:
        cr, cc = q.popleft()
        if not (0 <= cr < h and 0 <= cc < w) or out[cr][cc] != target:
            continue
        out[cr][cc] = new_color
        q.extend([(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)])
    return to_grid(out)


def fill_holes(grid, background=0, fill_color=None):
    """Fill background regions fully enclosed by non-background cells
    (flood fill from border to mark 'outside', everything else is a hole)."""
    h, w = dims(grid)
    outside = [[False] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] == background and not outside[r][c]:
                outside[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] == background and not outside[r][c]:
                outside[r][c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and grid[nr][nc] == background
                    and not outside[nr][nc]):
                outside[nr][nc] = True
                q.append((nr, nc))
    if fill_color is None:
        fill_color = most_common_color(grid, exclude_zero=True) or 1
    out = [list(row) for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == background and not outside[r][c]:
                out[r][c] = fill_color
    return to_grid(out)


# ---------------------------------------------------------------- registry for search

# name -> (fn, arity_beyond_grid). 0-arity ops are searched directly;
# parametrized ops get their param space enumerated by the caller.
UNARY_OPS = {
    "identity": identity,
    "rotate90": rotate90,
    "rotate180": rotate180,
    "rotate270": rotate270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
    "crop_to_content": crop_to_content,
    "fill_holes": fill_holes,
}

PARAM_OPS = {
    "recolor": recolor,          # needs mapping dict, built from task colors
    "tile": tile,                # needs (reps_r, reps_c)
    "scale": scale,              # needs (factor_r, factor_c)
    "downscale": downscale,      # needs (factor_r, factor_c)
    "symmetrize": symmetrize,    # needs axis
}

