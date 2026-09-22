"""Turn GhostBatchNorm's scalar *buffers* into Python scalars.

GhostBatchNorm1d registers its configuration as 0-dim tensors: ``ngb``, ``gbs``,
``eta``, ``eps`` and friends. That has two costs in the hot path, and there are
26 of these modules per FvT:

* ``self.gbs = batch_size // self.ngb`` yields a *tensor*, and
  ``x.view(self.ngb, self.gbs * pixels, ...)`` then calls ``__index__`` on it,
  i.e. ``.item()`` -- a device synchronisation. Measured at 226 ``aten::item``
  plus 226 ``aten::_local_scalar_dense`` per training step.
* ``self.eta * self.m`` and ``(gbv + self.eps).sqrt()`` mix tensors with tensor
  scalars, producing 539 ``aten::result_type`` promotion checks per step.

No ``forward`` rewrite is needed. Popping these out of ``_buffers`` and setting
them as plain attributes makes every use a Python scalar, which is what the
code already treats them as semantically.

**Caveat for production use.** Removing them from ``_buffers`` removes them from
``state_dict``, so ``load_state_dict(strict=True)`` would fail against the
121,595 existing ``*_best.pt`` checkpoints. For a benchmark that is irrelevant.
To adopt this for real, keep the buffers registered and add Python shadows that
``forward`` uses, refreshing them in ``_load_from_state_dict``.
"""

from __future__ import annotations

import torch

# scalars used only for shape arithmetic or as constants in the maths.
# m, s, m_biased and s_biased are genuine running statistics and stay tensors.
SCALAR_BUFFERS = (
    "ngb", "gbs", "eta", "eps", "bessel_correction",
    "t", "alpha", "beta1", "beta2",
)
INT_BUFFERS = {"ngb", "gbs"}


def patch_module(mod: torch.nn.Module) -> int:
    """Convert one module's scalar buffers in place. Returns how many."""
    n = 0
    buffers = mod._buffers
    for name in SCALAR_BUFFERS:
        if name not in buffers:
            continue
        val = buffers.pop(name)
        if val is None:
            continue
        py = int(val.item()) if name in INT_BUFFERS else float(val.item())
        object.__setattr__(mod, name, py)
        n += 1
    return n


def patch(model: torch.nn.Module) -> dict:
    """Convert every GhostBatchNorm-like module in a model."""
    converted = mods = 0
    for m in model.modules():
        if type(m).__name__.startswith("GhostBatchNorm"):
            c = patch_module(m)
            converted += c
            mods += c > 0
    return {"modules": mods, "buffers_converted": converted}
