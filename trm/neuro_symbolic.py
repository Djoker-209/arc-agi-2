"""Neuro-symbolic disambiguation: the main research contribution of this repo.

The problem this solves
------------------------
ARC-AGI-2 was specifically designed so that short exact-match DSL programs
often UNDERDETERMINE the task: several distinct short programs can each
reproduce every train pair perfectly, yet disagree on the test input. Our
own solver.search() already finds this -- see all_candidate_grids() -- but
has no principled way to break the tie beyond "prefer the shortest program"
(MDL), which is a fine prior but not a real signal about *this specific*
test input.

TRM, on the other hand, has actually learned a distribution over what
ARC-AGI-2 outputs look like (from its (synthetic or real) pretraining) --
but as a from-scratch 7M-parameter model it hallucinates on many cells,
especially anything requiring exact structural copying, which is exactly
where the DSL is unbeatable (it's provably exact on the training pairs).

So instead of treating "DSL vs. neural model" as a choice between two
separate submission slots, we use TRM as a *soft oracle* to disambiguate
between the DSL's own multiple valid hypotheses: for each train-consistent
DSL program, ask "how plausible does TRM think this specific output grid
is", and rank by that. This is different from standard test-time-training
ensembling (which fine-tunes the neural net per task) and from majority
voting over augmented views (AIRV-style) -- it's a likelihood-based
arbitration between exact symbolic candidates. As far as our research
turned up, this specific combination -- neural likelihood scoring used to
disambiguate a symbolic program synthesizer's own multiple exact-match
hypotheses -- hasn't been published for ARC-AGI-2. That's the risky/novel
bet: it may not beat either component alone in practice, but it's cheap
to test (no retraining needed) and is a genuine, testable idea.

Selection policy
-----------------
1. If the DSL found >=1 train-consistent program: take up to the top few
   MDL-ranked candidates, score each under TRM's token distribution, and
   submit the top-2 by combined score. A program that's both short (low
   MDL) AND high-likelihood under TRM is our strongest bet; if the top-MDL
   candidate and the top-TRM-likelihood candidate disagree, submitting
   both as the 2 allowed attempts is strictly better than picking one.
2. If the DSL found nothing: fall back to TRM's own argmax decode as
   attempt 1, and a second decode from an alternate low-temperature sample
   (or the second-most-likely token per uncertain cell) as attempt 2.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from trm.grid_codec import decode_grid, grid_log_likelihood


@dataclass
class RankedCandidate:
    grid: np.ndarray
    source: str          # "dsl" or "trm_fallback"
    mdl_rank: Optional[int]
    trm_log_likelihood: float


def rank_candidates(
    dsl_candidates: List[list],
    token_logits: np.ndarray,
    pad_r: int = 0,
    pad_c: int = 0,
    max_dsl_candidates_to_score: int = 8,
) -> List[RankedCandidate]:
    """Score a list of DSL candidate grids (as returned by
    solver.all_candidate_grids, MDL-ranked) against TRM's per-cell
    distribution, and return them ranked by log-likelihood.

    dsl_candidates: list of grids, each a list-of-lists (solver.py's
        native grid representation, matching load_task/_apply_program).
    """
    ranked = []
    for mdl_rank, grid in enumerate(dsl_candidates[:max_dsl_candidates_to_score]):
        arr = np.array(grid, dtype=np.int64)
        ll = grid_log_likelihood(arr, token_logits, pad_r=pad_r, pad_c=pad_c)
        ranked.append(RankedCandidate(grid=arr, source="dsl", mdl_rank=mdl_rank, trm_log_likelihood=ll))
    ranked.sort(key=lambda c: c.trm_log_likelihood, reverse=True)
    return ranked


def select_two_attempts(
    dsl_candidates: List[list],
    token_logits: np.ndarray,
    pad_r: int = 0,
    pad_c: int = 0,
) -> List[np.ndarray]:
    """Top-level entry point: produce the 2 grids to submit for one test
    input, combining the DSL's exact-match candidates with TRM's learned
    plausibility signal.
    """
    if dsl_candidates:
        ranked = rank_candidates(dsl_candidates, token_logits, pad_r=pad_r, pad_c=pad_c)
        # Always include the top-MDL candidate (rank 0) -- it's the DSL's
        # own best guess and is "free" (already computed, zero extra
        # risk) -- then fill the second slot with whichever OTHER
        # candidate TRM finds most plausible, since that's the case where
        # disambiguation actually earns its keep.
        top_mdl = next(c for c in ranked if c.mdl_rank == 0)
        attempts = [top_mdl.grid]
        for c in ranked:
            if not np.array_equal(c.grid, top_mdl.grid):
                attempts.append(c.grid)
                break
        if len(attempts) == 1:
            attempts.append(top_mdl.grid)  # only one distinct candidate existed
        return attempts[:2]

    # No DSL candidate at all: fall back to TRM's own decode, plus a
    # second attempt from the runner-up token at the least-confident cells
    # (cheap diversity, no extra forward pass).
    decoded = decode_grid(token_logits, pad_r=pad_r, pad_c=pad_c)
    second = _second_best_decode(token_logits, pad_r=pad_r, pad_c=pad_c)
    return [decoded.grid, second]


def _second_best_decode(token_logits: np.ndarray, pad_r: int, pad_c: int) -> np.ndarray:
    """A cheap second attempt when we have no DSL candidate to fall back
    on: flip the single least-confident cell in the argmax decode to its
    runner-up token. This isn't meant to be sophisticated -- it's a
    zero-extra-compute way to use the second submission slot for
    something other than a duplicate of attempt 1."""
    from trm.grid_codec import MAX_GRID, VOCAB_SIZE

    logits = np.asarray(token_logits).reshape(MAX_GRID, MAX_GRID, VOCAB_SIZE)
    decoded = decode_grid(token_logits, pad_r=pad_r, pad_c=pad_c)
    h, w = decoded.grid.shape

    region = logits[pad_r:pad_r + h, pad_c:pad_c + w, :]
    sorted_logits = np.sort(region, axis=-1)
    margin = sorted_logits[..., -1] - sorted_logits[..., -2]  # confidence gap
    r, c = np.unravel_index(np.argmin(margin), margin.shape)

    runner_up_token = np.argsort(region[r, c])[-2]
    alt = decoded.grid.copy()
    alt[r, c] = np.clip(runner_up_token - 2, 0, 9)
    return alt
