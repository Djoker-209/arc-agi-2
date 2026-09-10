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
    batch_size=1,
    seq_len=SEQ_LEN,
    puzzle_emb_ndim=512,
    num_puzzle_identifiers=1,   # single-task inference: one blank identifier is enough
    vocab_size=VOCAB_SIZE,
    H_cycles=3,
    L_cycles=6,
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
            # DDP/torchrun checkpoints are saved with a "module." prefix.
            state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                          for k, v in state_dict.items()}
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[trm] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys "
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
