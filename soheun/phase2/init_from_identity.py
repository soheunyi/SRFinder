"""Phase 2: initialize estimators from identity rather than stack position.

The current path is

    pl.seed_everything(model_seed)
    StackedFvTClassifier(...)      # builds num_stacks classifiers in a loop

so classifier *i* consumes whatever the global RNG holds after *i*
constructions. This module builds each estimator under a generator seeded from
its own ``EstimatorIdentity``, which makes the result independent of how many
estimators precede it and of how the stack is ordered.

Nothing here edits production code or changes any existing signature. It is
applied to an already-constructed ``StackedFvTClassifier`` in place.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Sequence

import torch

from fvt_classifier import FvTClassifier

from identity import EstimatorIdentity


@contextlib.contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    """Run a block under a fixed global torch seed, then restore the caller's.

    Restoring matters: without it, re-initializing a stack would shift the RNG
    that the training-order sampler later draws from, and a supposedly
    position-independent change would leak back into batch order.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_estimator(
    hparams: dict[str, Any],
    run_name: str,
    seed: int,
    device: str | torch.device = "cpu",
) -> FvTClassifier:
    """Construct one ``FvTClassifier`` under an isolated, explicit seed."""
    with isolated_rng(seed):
        model = FvTClassifier(
            num_classes=2,
            dim_input_jet_features=4,
            dim_dijet_features=int(hparams["dim_dijet_features"]),
            dim_quadjet_features=int(hparams["dim_quadjet_features"]),
            run_name=run_name,
            device=device,
            depth=hparams["depth"],
            repr_norm=bool(hparams.get("repr_norm", False)),
        )
    return model


def initial_state_for(
    identity: EstimatorIdentity,
    hparams: dict[str, Any],
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """The initial parameters an estimator has by virtue of *being that
    estimator*, with no reference to any stack."""
    seed = identity.seed("model_init")
    model = build_estimator(hparams, identity.fingerprint, seed, device=device)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def apply_identities(
    stacked_model,
    identities: Sequence[EstimatorIdentity],
    hparams_list: Sequence[dict[str, Any]],
) -> list[int]:
    """Re-initialize each member of a stack from its own identity, in place.

    Returns the initialization seed used for each position, so a run can record
    which estimator sat where without that placement affecting anything.
    """
    if len(identities) != len(stacked_model.fvt_classifiers):
        raise ValueError(
            f"{len(identities)} identities for "
            f"{len(stacked_model.fvt_classifiers)} estimators"
        )
    if len(hparams_list) != len(identities):
        raise ValueError("hparams_list and identities must be the same length")

    device = next(stacked_model.parameters()).device
    seeds: list[int] = []
    for i, identity in enumerate(identities):
        state = initial_state_for(identity, hparams_list[i], device=device)
        stacked_model.fvt_classifiers[i].load_state_dict(state)
        seeds.append(identity.seed("model_init"))
    return seeds
