"""Phase 2 gate: the three checkboxes from issue #2.

    [ ] Every estimator has an explicit initialization seed.
    [ ] Stack permutation does not change estimator initialization.
    [ ] Repeating a group reproduces its initial state exactly.

Runnable two ways:

    python phase2/test_identity.py          # prints a report, exits nonzero on failure
    pytest phase2/test_identity.py

Everything runs on CPU against freshly constructed models. No training, no data
loading, no GPU, seconds not minutes.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SOHEUN = HERE.parent
for p in (str(SOHEUN), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(SOHEUN)

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from stacked_fvt import StackedFvTClassifier  # noqa: E402

from identity import (  # noqa: E402
    PURPOSES,
    EstimatorIdentity,
    derive_seed,
    group_fingerprint,
    identity_from_step3_config,
)
from init_from_identity import apply_identities  # noqa: E402

CONFIG_PATTERN = (
    "configs/tmp/CR_fvt_training_ensemble_max_{seed}_0.0_0_0.1_0.2_0.8.yml"
)
GROUP_SIZE = 100
STACK_SIZE = 5


# --------------------------------------------------------------------- helpers


def load_group(n: int = GROUP_SIZE) -> list[dict]:
    configs = []
    for s in range(n):
        path = SOHEUN / CONFIG_PATTERN.format(seed=s)
        with open(path) as f:
            configs.append(yaml.safe_load(f))
    return configs


def state_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    sd = module.state_dict()
    for key in sorted(sd):
        v = sd[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def build_stack(hparams_list: list[dict], run_names: list[str]):
    """Construct a stack exactly the way train_stacked_fvt does."""
    pl.seed_everything(int(hparams_list[0]["model_seed"]), workers=True)
    return StackedFvTClassifier(
        num_stacks=len(hparams_list),
        num_classes=2,
        dim_input_jet_features=4,
        dim_dijet_features=int(hparams_list[0]["dim_dijet_features"]),
        dim_quadjet_features=int(hparams_list[0]["dim_quadjet_features"]),
        run_names=run_names,
        stacked_run_name="phase2_test",
        device="cpu",
        depth=hparams_list[0]["depth"],
        repr_norm=bool(hparams_list[0].get("repr_norm", False)),
    )


def digests_by_identity(
    identities: list[EstimatorIdentity], hparams_list: list[dict]
) -> dict[str, str]:
    """Build a stack in the given order, apply identities, digest each member."""
    run_names = [i.fingerprint for i in identities]
    stack = build_stack(hparams_list, run_names)
    apply_identities(stack, identities, hparams_list)
    return {
        identities[k].fingerprint: state_digest(stack.fvt_classifiers[k])
        for k in range(len(identities))
    }


def baseline_digests_by_identity(
    identities: list[EstimatorIdentity], hparams_list: list[dict]
) -> dict[str, str]:
    """Same, but WITHOUT applying identities: the current production behaviour."""
    run_names = [i.fingerprint for i in identities]
    stack = build_stack(hparams_list, run_names)
    return {
        identities[k].fingerprint: state_digest(stack.fvt_classifiers[k])
        for k in range(len(identities))
    }


# ----------------------------------------------------------------------- tests


def test_seed_derivation_is_deterministic_and_domain_separated():
    ident = EstimatorIdentity(
        experiment_name="x",
        step=3,
        mother_sample_seed=7,
        model_init_seed=0,
        training_order_seed=0,
        data_split_seed=0,
        ensemble_member_seed=0,
        smearing_noise_seed=0,
    )
    for p in PURPOSES:
        assert derive_seed(ident, p) == derive_seed(ident, p)
    seeds = {derive_seed(ident, p) for p in PURPOSES}
    assert len(seeds) == len(PURPOSES), "purposes must not collide"

    # every field participates
    import dataclasses

    base = derive_seed(ident, "model_init")
    for field in dataclasses.fields(ident):
        if field.name in ("experiment_name", "upstream_fingerprint"):
            bumped = dataclasses.replace(ident, **{field.name: "other"})
        elif field.name == "step":
            bumped = dataclasses.replace(ident, step=4)
        else:
            bumped = dataclasses.replace(
                ident, **{field.name: getattr(ident, field.name) + 1}
            )
        assert derive_seed(bumped, "model_init") != base, field.name


def test_every_estimator_has_an_explicit_initialization_seed():
    configs = load_group()
    identities = [identity_from_step3_config(c) for c in configs]

    assert len(identities) == GROUP_SIZE
    assert len({i.fingerprint for i in identities}) == GROUP_SIZE, (
        "identities must be distinct"
    )
    seeds = [i.seed("model_init") for i in identities]
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == GROUP_SIZE, "initialization seeds must be distinct"

    # the group is pinned by content, not by listing order
    assert group_fingerprint(identities) == group_fingerprint(list(reversed(identities)))


def test_stack_permutation_does_not_change_initialization():
    configs = load_group(STACK_SIZE)
    identities = [identity_from_step3_config(c) for c in configs]
    hparams = [_hparams(c) for c in configs]

    forward = digests_by_identity(identities, hparams)

    order = list(reversed(range(STACK_SIZE)))
    reversed_ids = [identities[k] for k in order]
    reversed_hp = [hparams[k] for k in order]
    backward = digests_by_identity(reversed_ids, reversed_hp)

    assert forward == backward, "initialization must not depend on stack position"


def test_repeating_a_group_reproduces_initial_state():
    configs = load_group(STACK_SIZE)
    identities = [identity_from_step3_config(c) for c in configs]
    hparams = [_hparams(c) for c in configs]

    first = digests_by_identity(identities, hparams)
    second = digests_by_identity(identities, hparams)
    assert first == second


def test_current_code_does_depend_on_stack_position():
    """Records the defect Phase 2 removes. If this ever stops holding, the
    production initialization path changed and Phase 2's premise needs a
    re-read."""
    configs = load_group(STACK_SIZE)
    identities = [identity_from_step3_config(c) for c in configs]
    hparams = [_hparams(c) for c in configs]

    forward = baseline_digests_by_identity(identities, hparams)

    order = list(reversed(range(STACK_SIZE)))
    backward = baseline_digests_by_identity(
        [identities[k] for k in order], [hparams[k] for k in order]
    )
    assert forward != backward, (
        "expected the unpatched path to be position dependent"
    )


def _hparams(config: dict) -> dict:
    """The subset of CR_fvt needed to construct a classifier."""
    cr = dict(config["CR_fvt"])
    return cr


# ------------------------------------------------------------------ standalone

TESTS = [
    ("seed derivation is deterministic and domain separated",
     test_seed_derivation_is_deterministic_and_domain_separated),
    ("every estimator has an explicit initialization seed",
     test_every_estimator_has_an_explicit_initialization_seed),
    ("stack permutation does not change initialization",
     test_stack_permutation_does_not_change_initialization),
    ("repeating a group reproduces its initial state",
     test_repeating_a_group_reproduces_initial_state),
    ("(baseline) current code DOES depend on stack position",
     test_current_code_does_depend_on_stack_position),
]


def main() -> int:
    import logging

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    print("Phase 2 gate: deterministic estimator identities\n")
    ok = True
    for name, fn in TESTS:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            ok = False
            print(f"  FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")

    print()
    configs = load_group(3)
    for c in configs:
        ident = identity_from_step3_config(c)
        print(f"  mother_sample_seed={ident.mother_sample_seed:<3} "
              f"fingerprint={ident.fingerprint[:16]}  "
              f"model_init_seed={ident.seed('model_init')}")
    print()
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
