"""Test-time adaptation for TRM.

Why this exists
----------------
TRM's puzzle_emb is a *sparse lookup table*, one row per training-time
task instance (the seconds-0/trm-arc2-8gpu checkpoint has 1,045,839 rows
-- roughly 1000 base tasks x ~1000 augmentations). There is no way to
recover which row (if any) corresponds to one of OUR tasks, and it
wouldn't help if we could: this architecture's whole approach to
generalizing to a NEW task is to fit a fresh embedding to it via gradient
descent on that task's own train pairs, not to look one up. This is
standard for this model family, not something we're bolting on -- but a
plain frozen forward pass (which is what a naive port of the reference
inference code does) skips it, which is almost certainly why published
low-effort scores for public checkpoints look close to random.

What this does
---------------
For one task: replace the model's puzzle_emb with a fresh, small,
trainable embedding (LearnablePuzzleEmbedding), freeze every other
parameter, and run a short Adam loop minimizing cross-entropy between
the model's predicted per-cell tokens and each train pair's actual output
grid (encoded at the SAME canvas offset as its input, since the model
predicts per-position, not autoregressively). Then run one more forward
pass on the test input using the now-adapted embedding.

This intentionally does NOT replicate the checkpoint's original
CastedSparseEmbeddingSignSGD_Distributed optimizer -- that exists purely
to make sparse updates efficient across many GPUs and millions of rows.
For one task and one embedding row, plain Adam on a real nn.Parameter is
simpler and exactly equivalent in spirit.
"""
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from trm.grid_codec import encode_grid, SEQ_LEN


class LearnablePuzzleEmbedding(nn.Module):
    """Drop-in replacement for CastedSparseEmbedding, but for exactly one
    task: a single trainable row, returned regardless of the identifier
    passed in (we only ever have one task loaded at a time)."""

    def __init__(self, puzzle_emb_ndim: int, init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, puzzle_emb_ndim) * init_std)

    def forward(self, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        return self.weight.expand(puzzle_identifiers.shape[0], -1)


def adapt_and_predict(
    runner,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_grid: np.ndarray,
    num_steps: int = 200,
    lr: float = 1e-2,
) -> np.ndarray:
    """Fine-tune a fresh puzzle embedding on this task's train pairs, then
    return (seq_len, vocab_size) logits for the test grid.

    train_pairs: list of (input_grid, output_grid) numpy arrays for this
        one task (its actual ARC train examples).
    """
    model = runner.model
    device = runner.device

    # Swap in a fresh, trainable embedding for THIS task only.
    original_puzzle_emb = model.inner.puzzle_emb
    model.inner.puzzle_emb = LearnablePuzzleEmbedding(model.config.puzzle_emb_ndim).to(device)

    # Freeze everything except the new embedding -- we're doing per-task
    # embedding adaptation, not fine-tuning the whole network.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.inner.puzzle_emb.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.inner.puzzle_emb.parameters(), lr=lr)

    # Encode train pairs: input and output MUST share the same canvas
    # offset (pad_r, pad_c) so that a prediction at position i is
    # compared against the correct target token at position i.
    input_batches, target_batches = [], []
    for inp, out in train_pairs:
        inp = np.asarray(inp)
        input_tokens = encode_grid(inp, pad_r=0, pad_c=0)
        target_tokens = encode_grid(np.asarray(out), pad_r=0, pad_c=0)
        input_batches.append(input_tokens)
        target_batches.append(target_tokens)
    inputs = torch.from_numpy(np.stack(input_batches)).long().to(device)
    targets = torch.from_numpy(np.stack(target_batches)).long().to(device)
    n = inputs.shape[0]

    model.train()
    try:
        for step in range(num_steps):
            batch = {
                "inputs": inputs,
                "puzzle_identifiers": torch.zeros(n, dtype=torch.long, device=device),
            }
            carry = model.initial_carry(batch)
            optimizer.zero_grad()
            # One refinement pass is enough per adaptation step; the model
            # itself already runs H_cycles*L_cycles internally per call.
            carry, outputs = model(carry, batch)
            logits = outputs["logits"]  # (n, seq_len, vocab_size)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()
    finally:
        model.eval()

    # Predict on the test input using the now-adapted embedding.
    with torch.no_grad():
        test_tokens = encode_grid(np.asarray(test_grid), pad_r=0, pad_c=0)
        batch = {
            "inputs": torch.from_numpy(test_tokens).long().unsqueeze(0).to(device),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device),
        }
        carry = model.initial_carry(batch)
        for _ in range(runner.config["halt_max_steps"]):
            carry, outputs = model(carry, batch)
            if bool(carry.halted[0]):
                break
        test_logits = outputs["logits"][0].float().cpu().numpy()

    # Restore the original (checkpoint's) puzzle_emb module so the runner
    # can be reused cleanly for the next task.
    model.inner.puzzle_emb = original_puzzle_emb
    for p in model.parameters():
        p.requires_grad = False

    return test_logits
