"""
Synthetic sanity checks for the TRM integration: grid codec round-trip,
DSL multi-candidate enumeration, and neuro-symbolic ranking/fallback.
No real TRM checkpoint is available in this sandbox (no internet access
to Kaggle/HuggingFace), so these tests use a tiny randomly-initialized
model just to prove the plumbing (shapes, decode, ranking logic) is
correct -- accuracy from a random-init model is meaningless and not
what's being checked here.
"""
import numpy as np

from trm.grid_codec import encode_grid, decode_grid, grid_log_likelihood, SEQ_LEN, VOCAB_SIZE
from trm.neuro_symbolic import select_two_attempts, rank_candidates
from trm.inference import TRMRunner, DEFAULT_CONFIG
import solver
from dsl import to_grid

results = []

TINY_CONFIG = dict(DEFAULT_CONFIG)
TINY_CONFIG.update(hidden_size=32, num_heads=2, H_cycles=1, L_cycles=1,
                    halt_max_steps=1, puzzle_emb_ndim=32)


def check(name, ok):
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    results.append(ok)
    return ok


# 1) codec round-trip on a real-shaped grid
grid = np.array([[1, 2, 3], [4, 5, 6]])
tokens = encode_grid(grid)
check("codec_shape", tokens.shape == (SEQ_LEN,))

# 2) decode recovers the exact grid from "perfect" one-hot logits
fake_logits = np.full((SEQ_LEN, VOCAB_SIZE), -10.0)
fake_logits[np.arange(SEQ_LEN), tokens] = 10.0
decoded = decode_grid(fake_logits)
check("decode_roundtrip", np.array_equal(decoded.grid, grid))

# 3) log-likelihood ranks the true grid above a perturbed one
wrong = grid.copy()
wrong[0, 0] = (wrong[0, 0] + 1) % 10
ll_correct = grid_log_likelihood(grid, fake_logits)
ll_wrong = grid_log_likelihood(wrong, fake_logits)
check("likelihood_prefers_correct", ll_correct > ll_wrong)

# 4) rank_candidates puts the TRM-favored grid first, even if it's not
# the lowest-MDL (rank 0) candidate -- this is the actual disambiguation
# behavior the whole module exists for.
low_mdl_grid = wrong        # pretend this was MDL-rank 0 (shorter program)
high_ll_grid = grid         # pretend this was MDL-rank 1 (longer program)
ranked = rank_candidates(
    [low_mdl_grid.tolist(), high_ll_grid.tolist()],
    fake_logits,
)
check("ranking_overrides_mdl_when_trm_disagrees", np.array_equal(ranked[0].grid, grid))

# 5) select_two_attempts always includes the top-MDL candidate plus the
# best alternative, never two copies when >=2 distinct candidates exist
attempts = select_two_attempts([low_mdl_grid.tolist(), high_ll_grid.tolist()], fake_logits)
check("two_distinct_attempts", len(attempts) == 2 and not np.array_equal(attempts[0], attempts[1]))

# 6) fallback path: no DSL candidates at all -> TRM decode + a distinct
# second attempt, still shaped correctly
attempts = select_two_attempts([], fake_logits)
check("fallback_shapes", all(a.shape == grid.shape for a in attempts))
check("fallback_attempt0_matches_trm_argmax", np.array_equal(attempts[0], decoded.grid))

# 7) end-to-end: TRMRunner + solver on a real synthetic ARC-style task
# (grids must be dsl.to_grid()'s tuple-of-tuples form -- solver.py's
# native representation -- not raw lists)
runner = TRMRunner(config=TINY_CONFIG).load()
train = [(to_grid([[1, 2], [3, 4]]), to_grid([[2, 1], [4, 3]])),
         (to_grid([[5, 6], [7, 8]]), to_grid([[6, 5], [8, 7]]))]
test_input = to_grid([[9, 0], [1, 2]])
programs = solver.search(train, max_depth=2)
dsl_candidates, _ = solver.all_candidate_grids(train, test_input, programs=programs)
check("e2e_dsl_finds_flip_h", len(programs) >= 1)
logits = runner.predict_logits(np.array(test_input))
attempts = select_two_attempts(dsl_candidates, logits)
check("e2e_correct_shape", attempts[0].shape == (2, 2))
check("e2e_correct_answer_present", any(np.array_equal(a, [[0, 9], [2, 1]]) for a in attempts))

print(f"\n{sum(results)}/{len(results)} TRM pipeline checks passed")

# --- test_time_adapt.py: verify the adaptation loop's plumbing is
# correct (loss decreases, shapes are right). NOT a convergence/accuracy
# check -- with the reasoning layers randomly initialized (no real
# checkpoint available in this sandbox), there's no reason to expect
# exact-match convergence; that's expected to improve substantially once
# real pretrained weights are loaded, since only the embedding is random
# in that case, not the whole network.
print()
from trm.test_time_adapt import LearnablePuzzleEmbedding
import torch

model2 = TRMRunner(config=TINY_CONFIG).load().model
train_pairs = [(np.array([[1, 2], [3, 4]]), np.array([[2, 1], [4, 3]])),
               (np.array([[5, 6], [7, 8]]), np.array([[6, 5], [8, 7]]))]
model2.inner.puzzle_emb = LearnablePuzzleEmbedding(model2.config.puzzle_emb_ndim)
for p in model2.parameters():
    p.requires_grad = False
for p in model2.inner.puzzle_emb.parameters():
    p.requires_grad = True
opt = torch.optim.Adam(model2.inner.puzzle_emb.parameters(), lr=0.02)
inputs = torch.from_numpy(np.stack([encode_grid(i) for i, o in train_pairs])).long()
targets = torch.from_numpy(np.stack([encode_grid(o) for i, o in train_pairs])).long()
model2.train()
losses = []
for _ in range(15):
    batch = {"inputs": inputs, "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    carry = model2.initial_carry(batch)
    opt.zero_grad()
    carry, outputs = model2(carry, batch)
    loss = torch.nn.functional.cross_entropy(
        outputs["logits"].reshape(-1, outputs["logits"].shape[-1]), targets.reshape(-1))
    loss.backward()
    opt.step()
    losses.append(loss.item())
check("ttt_loss_decreases", losses[-1] < losses[0])
check("ttt_only_embedding_has_grad", model2.inner.puzzle_emb.weight.requires_grad
      and not next(model2.inner.L_level.parameters()).requires_grad)

print(f"\n{sum(results)}/{len(results)} total checks passed")
