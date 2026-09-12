"""
batch_eval.py — Run the DSL solver (with and without dihedral-view voting)
over an entire directory of ARC-AGI-2 task JSON files and report accuracy.

Usage:
    python batch_eval.py data/ARC-AGI-2-data/data/evaluation
    python batch_eval.py data/ARC-AGI-2-data/data/training --limit 100
    python batch_eval.py data/ARC-AGI-2-data/data/evaluation --voting

Notes:
  - Public evaluation tasks include ground-truth "output" for test cases,
    so accuracy here is a real (if small/public) number, not a proxy.
  - This can be slow at max_depth=2 on 30x30 grids across 1000 training
    tasks; start with --limit for a quick read, then run the full set.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dsl
import solver
import augment
import symmetry_fill


def load_all_tasks(data_dir, limit=None):
    paths = sorted(Path(data_dir).glob("*.json"))
    if limit:
        paths = paths[:limit]
    return paths


def evaluate(data_dir, use_voting=False, limit=None, max_depth=2, verbose=False,
             trm_checkpoint=None, trm_device="cpu", trm_ttt_steps=200, trm_ttt_lr=1e-2):
    paths = load_all_tasks(data_dir, limit)
    n_tasks = len(paths)
    n_solved_program = 0   # search found >=1 fitting program
    n_correct = 0          # ground truth in top-2 candidates
    n_test_cases = 0
    n_sym_fired = 0        # symmetry_fill produced a confident answer
    n_sym_errors = 0       # symmetry_fill raised an exception
    failures = []
    start = time.time()

    trm_runner = None
    if trm_checkpoint is not None:
        from trm.inference import TRMRunner
        trm_runner = TRMRunner(checkpoint_path=trm_checkpoint, device=trm_device).load()
        if verbose:
            print(f"  Loaded TRM checkpoint from {trm_checkpoint} onto {trm_device}")

    for i, path in enumerate(paths):
        train, test_inputs, test_out = solver.load_task(str(path))
        if test_out is None:
            continue  # no ground truth to score against (shouldn't happen for eval set)

        try:
            sym_preds = symmetry_fill.solve(train, test_inputs)
        except Exception as e:
            sym_preds = []
            n_sym_errors += 1
            if verbose:
                print(f"  [{path.name}] symmetry_fill ERROR: {e}")

        try:
            if trm_runner is not None:
                # Neuro-symbolic mode: DSL finds all train-consistent
                # candidates, TRM's learned distribution disambiguates
                # between them (or supplies a fallback when the DSL
                # finds none). See trm/neuro_symbolic.py for the rationale.
                from trm.neuro_symbolic import select_two_attempts
                from trm.test_time_adapt import adapt_and_predict
                programs = solver.search(train, max_depth=max_depth)
                if programs:
                    n_solved_program += 1
                # Adapt once per task (uses the task's own train pairs),
                # not once per test input -- test_time_adapt.py trains a
                # fresh puzzle embedding for this task, which is shared
                # across all of the task's test inputs.
                train_np = [(np.array(i), np.array(o)) for i, o in train]
                preds = []
                for grid in test_inputs:
                    dsl_candidates, _ = solver.all_candidate_grids(train, grid, programs=programs)
                    logits = adapt_and_predict(trm_runner, train_np, np.array(grid),
                                                num_steps=trm_ttt_steps, lr=trm_ttt_lr)
                    attempts = select_two_attempts(dsl_candidates, logits)
                    preds.append([a.tolist() for a in attempts])
            elif use_voting:
                preds, _ = augment.solve_with_voting(train, test_inputs, max_depth=max_depth)
            else:
                preds, programs = solver.solve_task(train, test_inputs, max_depth=max_depth)
                if programs:
                    n_solved_program += 1
        except Exception as e:
            preds = [[g] for g in test_inputs]  # fallback: guess identity
            if verbose:
                print(f"  [{path.name}] ERROR during solve: {e}")

        # symmetry_fill only fires when it exactly fits every train pair,
        # so when it has an answer, trust it over the general DSL search.
        if sym_preds and all(sym_preds):
            n_sym_fired += 1
            preds = sym_preds

        for gt, cands in zip(test_out, preds):
            n_test_cases += 1
            if gt in cands:
                n_correct += 1
            else:
                failures.append(path.name)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  ...{i + 1}/{n_tasks} tasks, "
                  f"{n_correct}/{n_test_cases} test cases correct so far, "
                  f"{elapsed:.1f}s elapsed")

    elapsed = time.time() - start
    return {
        "n_tasks": n_tasks,
        "n_test_cases": n_test_cases,
        "n_correct": n_correct,
        "n_solved_program": n_solved_program,
        "n_sym_fired": n_sym_fired,
        "n_sym_errors": n_sym_errors,
        "accuracy": n_correct / n_test_cases if n_test_cases else 0.0,
        "elapsed_sec": elapsed,
        "failures": failures,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Directory of ARC-AGI-2 task JSON files, "
                                      "e.g. data/ARC-AGI-2-data/data/evaluation")
    ap.add_argument("--voting", action="store_true",
                     help="Use dihedral-view voting (augment.py) instead of plain search")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only evaluate the first N tasks (useful for a quick check)")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    ap.add_argument("--trm-checkpoint", default=None,
                     help="Path to a TRM checkpoint (.pt state_dict). When set, uses the "
                          "neuro-symbolic ranker (trm/neuro_symbolic.py) instead of plain "
                          "DSL search or voting -- overrides --voting.")
    ap.add_argument("--trm-device", default="cpu", help="cpu or cuda")
    ap.add_argument("--trm-ttt-steps", type=int, default=200,
                     help="Adam steps for per-task puzzle-embedding adaptation (test_time_adapt.py)")
    ap.add_argument("--trm-ttt-lr", type=float, default=1e-2)
    args = ap.parse_args()

    mode = "TRM neuro-symbolic" if args.trm_checkpoint else ("voting" if args.voting else "plain search")
    print(f"Evaluating {mode} solver on {args.data_dir} (max_depth={args.max_depth})...")
    results = evaluate(args.data_dir, use_voting=args.voting, limit=args.limit,
                        max_depth=args.max_depth, verbose=not args.quiet,
                        trm_checkpoint=args.trm_checkpoint, trm_device=args.trm_device,
                        trm_ttt_steps=args.trm_ttt_steps, trm_ttt_lr=args.trm_ttt_lr)

    print()
    print(f"Tasks evaluated:      {results['n_tasks']}")
    print(f"Test cases scored:    {results['n_test_cases']}")
    print(f"Correct:              {results['n_correct']}")
    print(f"Accuracy:             {results['accuracy']*100:.2f}%")
    print(f"symmetry_fill fired:  {results['n_sym_fired']}/{results['n_tasks']} tasks "
          f"({results['n_sym_errors']} errors)")
    if not args.voting:
        print(f"Tasks w/ any program: {results['n_solved_program']}/{results['n_tasks']} "
              f"(program found but possibly wrong on test)")
    print(f"Time elapsed:         {results['elapsed_sec']:.1f}s")
