# ARC-AGI-2 Baseline

Baseline solver for the [ARC Prize 2026 – ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) Kaggle track.

## Approach

Brute-force program synthesis over a small DSL of grid primitives
(rotations, flips, recoloring, cropping, tiling, scaling, symmetry
completion, connected-component/object extraction, flood fill). For each
task, the solver searches depth-1 and depth-2 compositions of these
primitives, plus parameter-inferring ops (recolor mapping, scale factor)
derived from the train pairs, and keeps any program that exactly
reproduces every train pair. Up to 2 candidate outputs are returned per
test input, matching the competition's scoring rule.

This is a baseline, not the final approach — it only solves tasks
expressible as a short exact composition of these primitives. ARC-AGI-2
was designed to resist exactly this class of solver. Next step: test-time
adaptation / fine-tuning per task, which has produced the largest score
jumps in past ARC Prize cycles.

## Files

- `dsl.py` — grid transformation primitives
- `solver.py` — task loader + composition search + CLI (`python solver.py path/to/task.json`)
- `test_solver.py` — synthetic sanity tests (no real task data required)
- `data/` — put ARC-AGI-2 task JSON here (not included; download from Kaggle or [github.com/arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2))

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
python test_solver.py      # should print "6/6 synthetic tasks passed"
```

