"""Load a TRM checkpoint and run inference for one task.

Config values below (hidden_size=512, H_cycles=3, L_cycles=6, ...) match
the published `arch=trm` config in SamsungSAILMontreal/TinyRecursiveModels
(config/arch/trm.yaml), which is what the NVARC / cpmpml public ARC-AGI-2
checkpoints were trained with. If you load a checkpoint trained with a
different config, override these to match -- a shape mismatch on load
means the config is wrong, not that the checkpoint is bad.

This module does NOT ship trained weights (none are downloadable from
this environment -- no internet access to Kaggle/HuggingFace here). See
README_TRM.md for how to fetch a public checkpoint on Kaggle and point
this loader at it.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from trm.model.trm import TinyRecursiveReasoningModel_ACTV1
from trm.grid_codec import SEQ_LEN, VOCAB_SIZE, encode_grid

DEFAULT_CONFIG = dict(
    # Matches the exact training command for seconds-0/trm-arc2-8gpu
    # (domus-magna/trm-repro): H_cycles=3, L_cycles=4, L_layers=2.
    # NOTE: the upstream repo's generic config/arch/trm.yaml default is
    # L_cycles=6 -- that is NOT what this specific checkpoint used. If
    # you load a different checkpoint, check ITS training command before
    # assuming these values -- a shape/key mismatch on load usually means
    # this config doesn't match how that particular checkpoint was
    # trained, not that the checkpoint file is corrupt.
    batch_size=1,
    seq_len=SEQ_LEN,
    puzzle_emb_ndim=512,
    num_puzzle_identifiers=1,   # single-task inference: puzzle_emb is replaced per-task by
                                # test_time_adapt.py, not looked up from the checkpoint's table
    vocab_size=VOCAB_SIZE,
    H_cycles=3,
    L_cycles=4,
    H_layers=0,
    L_layers=2,
    hidden_size=512,
    expansion=4,
    num_heads=8,
    pos_encodings="rope",
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    forward_dtype="float32",   # CPU-safe; use "bfloat16" only on GPU
)


@dataclass
class TRMRunner:
    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    config: dict = field(default_factory=lambda: dict(DEFAULT_CONFIG))
    model: Optional[torch.nn.Module] = field(default=None, init=False)

    def load(self):
        self.model = TinyRecursiveReasoningModel_ACTV1(self.config).to(self.device)
        if self.checkpoint_path is not None:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            # Checkpoints can stack multiple wrapper prefixes depending on
            # how they were saved: DDP ("module."), torch.compile
            # ("_orig_mod."), and/or a loss-head wrapper ("model."). Strip
            # all of them repeatedly until keys match our bare inner.* names.
            known_prefixes = ("module.", "_orig_mod.", "model.")
            def _strip(k):
                changed = True
                while changed:
                    changed = False
                    for p in known_prefixes:
                        if k.startswith(p):
                            k = k[len(p):]
                            changed = True
                return k
            state_dict = {_strip(k): v for k, v in state_dict.items()}
            # The checkpoint's puzzle_emb table is indexed by TRAINING-time
            # task IDs (often 1M+ rows -- one per augmented training
            # instance). We have no way to recover which row, if any,
            # corresponds to one of OUR tasks, and it wouldn't matter
            # anyway: this architecture is meant to be adapted per new
            # task via gradient descent (see test_time_adapt.py), not
            # looked up. Always skip it and start from a fresh, small
            # (num_puzzle_identifiers=1) embedding instead.
            state_dict = {k: v for k, v in state_dict.items() if "puzzle_emb" not in k}
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            missing = [m for m in missing if "puzzle_emb" not in m]
            if missing or unexpected:
                print(f"[trm] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys "
                      f"(excluding the intentionally-skipped puzzle_emb table) "
                      f"-- check DEFAULT_CONFIG matches the checkpoint's training config.")
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_logits(self, grid: np.ndarray, pad_r: int = 0, pad_c: int = 0) -> np.ndarray:
        """Run the full ACT loop (halt_max_steps refinement steps) on one
        grid and return (seq_len, vocab_size) logits for the final step.
        """
        assert self.model is not None, "call .load() first"
        tokens = encode_grid(grid, pad_r=pad_r, pad_c=pad_c)
        batch = {
            "inputs": torch.from_numpy(tokens).long().unsqueeze(0).to(self.device),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=self.device),
        }
        carry = self.model.initial_carry(batch)
        for _ in range(self.config["halt_max_steps"]):
            carry, outputs = self.model(carry, batch)
            if bool(carry.halted[0]):
                break
        return outputs["logits"][0].float().cpu().numpy()
