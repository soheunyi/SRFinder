"""Phase 2: immutable estimator identity.

Issue #2 asks for six separately recorded identity components, and for model
initialization to be derived from that identity rather than from a position in
the stack.

What the current code does instead
----------------------------------

``train_stacked_fvt`` calls ``pl.seed_everything(model_seed)`` once and then
constructs ``num_stacks`` classifiers in a loop, so estimator *i* gets whatever
the global RNG happens to hold after *i* constructions. For the eta=0.1 pilot
group every member carries ``model_seed = train_seed = data_seed = 0``, so
**position in the stack is the only thing that distinguishes one estimator's
initialization from another's**. Regrouping or reordering a campaign silently
re-initializes every estimator.

What this module provides
-------------------------

``EstimatorIdentity`` records the six components by name, plus the upstream
provenance that actually pins a step-3 CR estimator. ``derive_seed`` turns an
identity plus a purpose label into a stable 31-bit seed via BLAKE2b over a
canonical JSON encoding, so the value does not depend on Python's hash
randomization, dict ordering, interpreter version or machine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterable, Literal

Purpose = Literal["model_init", "training_order", "data_split", "smearing_noise"]

PURPOSES: tuple[Purpose, ...] = (
    "model_init",
    "training_order",
    "data_split",
    "smearing_noise",
)

_SEED_MODULUS = 2**31  # torch.manual_seed accepts well beyond this; stay small


@dataclasses.dataclass(frozen=True, order=True)
class EstimatorIdentity:
    """Immutable identity of one estimator.

    The six seed fields are the ones issue #2 asks to separate. They are kept
    distinct even where the current campaign happens to set them equal, so that
    disentangling them later is a config change rather than a code change.

    ``upstream_fingerprint`` pins the already-trained models a step-3 CR
    estimator depends on (the SR-stats ensemble). It is part of the identity
    because two CR estimators trained against different upstream ensembles are
    different estimators, even with identical seeds.
    """

    experiment_name: str
    step: int

    mother_sample_seed: int
    model_init_seed: int
    training_order_seed: int
    data_split_seed: int
    ensemble_member_seed: int
    smearing_noise_seed: int

    upstream_fingerprint: str = ""

    def canonical(self) -> str:
        """Stable JSON encoding. Field order is fixed by the dataclass."""
        return json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":")
        )

    @property
    def fingerprint(self) -> str:
        """Content-derived identity, unlike ``utils.create_hash``."""
        return hashlib.blake2b(self.canonical().encode(), digest_size=16).hexdigest()

    def seed(self, purpose: Purpose) -> int:
        return derive_seed(self, purpose)

    def seeds(self) -> dict[str, int]:
        return {p: derive_seed(self, p) for p in PURPOSES}


def derive_seed(identity: EstimatorIdentity, purpose: Purpose) -> int:
    """Stable seed for one purpose.

    Domain-separated by ``purpose`` so that the initialization seed, the
    training-order seed and the split seed of a single estimator are unrelated
    to each other. Independent of stack position by construction: nothing but
    the identity's own fields enters.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}, expected one of {PURPOSES}")
    payload = f"{purpose}\x00{identity.canonical()}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def upstream_fingerprint(hashes: Iterable[str]) -> str:
    """Order-insensitive digest of the upstream model hashes.

    Sorted first, because ``SR_stats_hashes`` is written in whatever order the
    metadata scan produced; the same ensemble listed in a different order is
    the same ensemble.
    """
    items = sorted(str(h) for h in hashes)
    h = hashlib.blake2b(digest_size=16)
    for item in items:
        h.update(item.encode())
        h.update(b"\x00")
    return h.hexdigest()


def identity_from_step3_config(config: dict[str, Any]) -> EstimatorIdentity:
    """Build an identity from one step-3 CR config as written on disk.

    Maps the six components onto the keys the campaign actually uses:

    ==========================  ===========================================
    identity component          source
    ==========================  ===========================================
    mother_sample_seed          ``dataset.seed``     (0..99, the subsample)
    model_init_seed             ``CR_fvt.model_seed``
    training_order_seed         ``CR_fvt.train_seed``
    data_split_seed             ``CR_fvt.data_seed``
    ensemble_member_seed        ``CR_fvt.ensemble_member`` if present, else 0
    smearing_noise_seed         ``smearing.seed`` if present, else 0
    upstream_fingerprint        digest of ``signal_region.SR_stats_hashes``
    ==========================  ===========================================

    ``smearing.seed`` is absent from step-3 configs (it lives in the step-2
    records the SR stats came from) and defaults to 0, which is the value every
    step-2 ensemble member in the eta=0.1 group carries.
    """
    cr = config["CR_fvt"]
    dataset = config.get("dataset", {})
    smearing = config.get("smearing", {})
    sr = config.get("signal_region", {})

    return EstimatorIdentity(
        experiment_name=str(config["experiment_name"]),
        step=int(config.get("step", 3)),
        mother_sample_seed=int(dataset["seed"]),
        model_init_seed=int(cr["model_seed"]),
        training_order_seed=int(cr["train_seed"]),
        data_split_seed=int(cr["data_seed"]),
        ensemble_member_seed=int(cr.get("ensemble_member", 0)),
        smearing_noise_seed=int(smearing.get("seed", 0)),
        upstream_fingerprint=upstream_fingerprint(sr.get("SR_stats_hashes", [])),
    )


def identity_from_hparams(hparams: dict[str, Any]) -> EstimatorIdentity:
    """Same mapping, but from a ``TrainingInfo.hparams`` dict.

    Step-3 hparams are the flattened ``CR_fvt`` block with ``experiment_name``,
    ``dataset``, ``signal_region`` and ``step`` grafted on by
    ``check_and_get_CR_fvt_hparams``.
    """
    dataset = hparams.get("dataset", {})
    sr = hparams.get("signal_region", {})
    smearing = hparams.get("smearing", {})

    return EstimatorIdentity(
        experiment_name=str(hparams["experiment_name"]),
        step=int(hparams.get("step", 3)),
        mother_sample_seed=int(dataset["seed"]),
        model_init_seed=int(hparams["model_seed"]),
        training_order_seed=int(hparams["train_seed"]),
        data_split_seed=int(hparams["data_seed"]),
        ensemble_member_seed=int(hparams.get("ensemble_member", 0)),
        smearing_noise_seed=int(smearing.get("seed", 0)),
        upstream_fingerprint=upstream_fingerprint(sr.get("SR_stats_hashes", [])),
    )


def group_fingerprint(identities: Iterable[EstimatorIdentity]) -> str:
    """Digest of a whole group, insensitive to the order the group is listed in.

    Two campaigns that train the same 100 estimators in different stack orders
    share a group fingerprint. That is the property Phase 2 is after.
    """
    items = sorted(i.fingerprint for i in identities)
    h = hashlib.blake2b(digest_size=16)
    for item in items:
        h.update(item.encode())
        h.update(b"\x00")
    return h.hexdigest()
