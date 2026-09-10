# TRM + neuro-symbolic disambiguation

## What's here

```
trm/
  model/           vendored from SamsungSAILMontreal/TinyRecursiveModels (MIT license)
    common.py, layers.py, sparse_embedding.py, trm.py
  grid_codec.py     grid <-> token conversion, EXACTLY matching the tokenization
                     the public checkpoints were trained with (see docstring)
  inference.py      TRMRunner: load a checkpoint, run the ACT loop, get logits
  neuro_symbolic.py the actual research contribution -- see below
test_trm_pipeline.py  synthetic sanity checks (10/10 passing, no real checkpoint needed)
```

`solver.py` gained one new function, `all_candidate_grids()`, additive and
non-breaking -- `solve_task()` and every existing caller are untouched.
`batch_eval.py` gained a `--trm-checkpoint PATH` flag.

## Why this exists (read this before assuming it'll just work)

The published TRM training recipe for ARC-AGI-2 assumes **4x H100 GPUs for
~3 days** (~288 GPU-hours). Your Kaggle submission notebook gets 12 hours
on L4x4. Pretraining TRM from scratch inside a submission notebook is not
happening -- not "tight", genuinely off by more than an order of
magnitude. This is normal: Kaggle submission notebooks are for
*inference*, not full training. The real workflow:

1. Get a **trained** TRM checkpoint. NVARC's actual 2025 winning entry is
   open-sourced (a condition of their prize) at
   `kaggle.com/code/cpmpml/arc2-trm-v31` and `github.com/arnuvt/NVARC/TRM`.
   You are allowed to build on it -- competition rules permit any freely,
   publicly available pretrained model.
2. Point `TRMRunner(checkpoint_path=...)` at it.
3. Everything in this repo (the ranker, the DSL, batch_eval) runs on top
   of that checkpoint's *inference*, which comfortably fits in 12 hours.

**I could not download real weights in the sandbox I built this in** --
no internet access to kaggle.com or huggingface.co from there. Every test
in `test_trm_pipeline.py` uses a randomly-initialized tiny model purely to
prove the plumbing (shapes, decode, ranking logic) is correct. Getting a
real checkpoint and re-running with it is the next real step, on your
machine or in a Kaggle notebook (where you do have internet, just not at
*evaluation* time).

### Getting the real checkpoint
```
# On Kaggle (has internet outside of the eval sandbox), or locally with
# kaggle CLI configured:
kaggle kernels output cpmpml/arc2-trm-v31 -p ./trm_checkpoint
# Look for a .pt / .pth file in the output -- that's your checkpoint_path.
```
If the state_dict's keys don't match on load, `TRMRunner.load()` prints
how many keys were missing/unexpected -- that means `DEFAULT_CONFIG` in
`trm/inference.py` doesn't match whatever config that checkpoint was
actually trained with. Check `arnuvt/NVARC/TRM` for their exact config
overrides and update `DEFAULT_CONFIG` to match; a shape mismatch means the
config is wrong, not that the checkpoint is broken.

## The research bet: neuro-symbolic disambiguation

This is the part that's actually novel, and actually risky -- it might
not beat either component alone.

**The problem:** ARC-AGI-2 is designed so several distinct short DSL
programs often fit every train pair perfectly, then disagree on the test
input. `solver.search()` already finds all of them, ranked by MDL (fewest
ops = most likely correct), but MDL alone doesn't know anything about
*this specific* test input -- it's a prior, not a fit-to-evidence check.

**The idea:** use TRM as a soft oracle. It has actually learned *some*
distribution over what ARC-AGI-2 outputs look like. For each DSL
candidate, ask "how plausible does TRM think this exact grid is", via
per-cell log-likelihood under TRM's predicted token distribution
(`grid_log_likelihood()` in `grid_codec.py`), and use that to pick between
otherwise-tied symbolic candidates, or as the tiebreaker for which 2 of
several MDL-tied candidates to actually submit.

This is different from standard TTT ensembling (which fine-tunes the
network per task) and from AIRV-style voting (which votes across
augmented views of the *same* model's own output). It's likelihood-based
arbitration between a *symbolic* synthesizer's own multiple exact-match
hypotheses. I didn't find this specific combination published for
ARC-AGI-2 in the research I did -- which is exactly why it's a bet, not a
guaranteed win. It costs nothing to test once you have a real checkpoint
(no retraining, just an extra forward pass per task), which is why it's
worth trying despite the risk.

### What to check once you have real weights
1. Run `batch_eval.py --trm-checkpoint <path> data/ARC-AGI-2-data/data/evaluation`
   and compare against the existing 0% plain-DSL baseline and TRM's own
   solo published 8%.
2. If neuro-symbolic beats both, that's your headline result for the
   Solution Writeup's Novelty criterion. If it doesn't, that's still a
   real, reportable negative result -- and cheap to have found out.
