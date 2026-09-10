"""Grid <-> token codec for TRM.

This MUST exactly match the tokenization scheme used to train the public
TRM checkpoints (see SamsungSAILMontreal/TinyRecursiveModels,
dataset/build_arc_dataset.py), or loaded weights will produce garbage:

    - vocab_size = 12: PAD=0, EOS=1, color c (0-9) -> token c+2
    - grids are placed inside a fixed 30x30 canvas, flattened row-major
      (seq_len = 900)
    - at train time the grid is placed at a random offset (pad_r, pad_c)
      as a translation augmentation; EOS tokens mark the row/col just
      past the real content when that boundary falls inside the canvas
    - everywhere else in the canvas is PAD

At inference time we don't need randomness: placing the grid at (0, 0)
is a valid, in-distribution input (translation augmentation covers all
offsets including the origin), and is what you want anyway since the
predicted output must be decoded back to an unambiguous grid.
"""
from dataclasses import dataclass

import numpy as np

MAX_GRID = 30
SEQ_LEN = MAX_GRID * MAX_GRID
VOCAB_SIZE = 12
PAD_TOKEN = 0
EOS_TOKEN = 1
COLOR_TOKEN_OFFSET = 2  # color c -> token c + 2


def encode_grid(grid: np.ndarray, pad_r: int = 0, pad_c: int = 0) -> np.ndarray:
    """Encode a (H, W) int grid (values 0-9) into a flat (900,) token array."""
    grid = np.asarray(grid)
    assert grid.ndim == 2, "grid must be 2D"
    nrow, ncol = grid.shape
    assert nrow <= MAX_GRID and ncol <= MAX_GRID, f"grid too large: {grid.shape}"
    assert pad_r + nrow <= MAX_GRID and pad_c + ncol <= MAX_GRID, "offset pushes grid out of canvas"

    canvas = np.full((MAX_GRID, MAX_GRID), PAD_TOKEN, dtype=np.int64)
    canvas[pad_r:pad_r + nrow, pad_c:pad_c + ncol] = grid + COLOR_TOKEN_OFFSET

    eos_row = pad_r + nrow
    eos_col = pad_c + ncol
    if eos_row < MAX_GRID:
        canvas[eos_row, pad_c:pad_c + ncol] = EOS_TOKEN
    if eos_col < MAX_GRID:
        canvas[pad_r:pad_r + nrow, eos_col] = EOS_TOKEN

    return canvas.reshape(-1)


@dataclass
class DecodedGrid:
    grid: np.ndarray            # (h, w) int array, colors 0-9
    cell_log_probs: np.ndarray  # (h, w) log P(argmax token) per cell, for confidence/ranking


def decode_grid(token_logits: np.ndarray, pad_r: int = 0, pad_c: int = 0) -> DecodedGrid:
    """Decode a (900, vocab_size) logit array back into a grid.

    Finds the EOS/PAD boundary from the argmax tokens to recover (h, w),
    then reads colors out of the (pad_r, pad_c) offset region. Returns
    per-cell log-probabilities too, since the disambiguation module needs
    them, not just the argmax grid.
    """
    logits = np.asarray(token_logits)
    assert logits.shape == (SEQ_LEN, VOCAB_SIZE), logits.shape

    log_probs = logits - _logsumexp(logits, axis=-1, keepdims=True)
    tokens = logits.argmax(axis=-1).reshape(MAX_GRID, MAX_GRID)

    # Find where the real content ends by scanning for the first EOS/PAD
    # row/col starting at the known offset.
    h = 0
    for r in range(pad_r, MAX_GRID):
        if tokens[r, pad_c] in (EOS_TOKEN, PAD_TOKEN):
            break
        h += 1
    w = 0
    for c in range(pad_c, MAX_GRID):
        if tokens[pad_r, c] in (EOS_TOKEN, PAD_TOKEN):
            break
        w += 1
    h = max(h, 1)
    w = max(w, 1)

    region = tokens[pad_r:pad_r + h, pad_c:pad_c + w]
    grid = np.clip(region - COLOR_TOKEN_OFFSET, 0, 9)

    log_probs_2d = log_probs.reshape(MAX_GRID, MAX_GRID, VOCAB_SIZE)
    cell_lp = log_probs_2d[pad_r:pad_r + h, pad_c:pad_c + w, :].max(axis=-1)

    return DecodedGrid(grid=grid.astype(np.int64), cell_log_probs=cell_lp)


def grid_log_likelihood(candidate: np.ndarray, token_logits: np.ndarray, pad_r: int = 0, pad_c: int = 0) -> float:
    """Score an externally-produced candidate grid (e.g. from the DSL
    solver) under TRM's predicted token distribution. This is the core
    primitive the neuro-symbolic ranker builds on: it lets us ask "how
    plausible does TRM think this DSL-derived grid is", without needing
    TRM's own argmax to be correct.

    Returns -inf if the candidate's shape doesn't fit at this offset.
    """
    candidate = np.asarray(candidate)
    h, w = candidate.shape
    if pad_r + h > MAX_GRID or pad_c + w > MAX_GRID:
        return float("-inf")

    logits = np.asarray(token_logits)
    log_probs = logits - _logsumexp(logits, axis=-1, keepdims=True)
    log_probs_2d = log_probs.reshape(MAX_GRID, MAX_GRID, VOCAB_SIZE)

    region = log_probs_2d[pad_r:pad_r + h, pad_c:pad_c + w, :]
    target_tokens = candidate + COLOR_TOKEN_OFFSET

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    cell_lp = region[rows, cols, target_tokens]
    return float(cell_lp.sum())


def _logsumexp(x: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)
