"""Standalone/group/reordered model-seed initialization for both architectures."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from member_initialization import initialize_members
from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier


def build(kind, name):
    if kind == 'fvt':
        return FvTClassifier(2,4,6,6,name,device='cpu',depth={'encoder':4,'decoder':1})
    return AttentionClassifier(6,2,name,depth=1)


def main():
    torch.set_num_threads(1)
    for kind in ('fvt','attention'):
        grouped = [build(kind,str(i)) for i in range(3)]
        before = torch.random.get_rng_state().clone()
        seeds = [3,17,29]
        initialize_members(grouped, [{'model_seed': s} for s in seeds])
        assert torch.equal(before,torch.random.get_rng_state()), 'Caller RNG changed'
        for pos, seed in enumerate(seeds):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                solo = build(kind, 'different-name')
            assert all(torch.equal(v, solo.state_dict()[k]) for k,v in grouped[pos].state_dict().items())
        reversed_group = [build(kind,str(i)) for i in range(3)]
        initialize_members(reversed_group, [{'model_seed': s} for s in reversed(seeds)])
        for a,b in zip(grouped,reversed(reversed_group)):
            assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        assert any(not torch.equal(a,b) for a,b in zip(grouped[0].parameters(),grouped[1].parameters()))
    from phase2.identity import identity_from_hparams
    hp = {'step': 3, 'experiment_name': 'test', 'dataset': {'seed': 9},
          'model_seed': 0, 'train_seed': 0, 'data_seed': 0,
          'signal_region': {'SR_stats_hashes': ['upstream_b','upstream_a']}}
    member = build('fvt', 'CR')
    record = initialize_members([member], [hp])[0]
    expected_seed = identity_from_hparams(hp).seed('model_init')
    assert record['model_seed'] == expected_seed
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(expected_seed)
        reference = build('fvt', 'reference')
    assert all(torch.equal(v, reference.state_dict()[k]) for k,v in member.state_dict().items())
    print('PASS: FvT/attention standalone, grouped, reordered seed initialization; caller RNG unchanged')


if __name__ == '__main__':
    main()
