"""Known-ratio and weight-scaling checks through the actual independent loss step.

A saturated four-bin classifier has an exact known population optimum. This
checks implementation semantics, not CR-to-SR extrapolation or signed-weight
neural-network consistency. No production recipe is modified.
"""
import argparse
import json
import pathlib
import sys
from types import SimpleNamespace
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from independent_training import step, epoch_losses


class BinClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_ratio = torch.nn.Parameter(torch.zeros(4))

    def forward(self, indices):
        value = self.log_ratio[indices]
        return torch.stack((torch.zeros_like(value), value), dim=1)


def wrapper(model, optimizer):
    host = SimpleNamespace(device=torch.device('cpu'), datamodule=SimpleNamespace(),
                           ce_loss=lambda logits, y: F.cross_entropy(logits, y, reduction='none'),
                           manual_backward=lambda loss: loss.backward(), optimizers=lambda: [optimizer])
    reset(host)
    return host


def reset(host):
    for name in ('losses_per_stack', 'weights_per_stack', 'batch_sizes'):
        setattr(host, 'train_' + name, torch.empty(0))


def learn(prior_ratio, common_scale):
    p3 = torch.tensor([.1, .2, .3, .4])
    p4 = torch.tensor([.4, .3, .2, .1])
    x = torch.arange(4).repeat(2)
    y = torch.tensor([0]*4+[1]*4)
    w = torch.cat((p3, prior_ratio*p4))*common_scale
    model = BinClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=.01, eps=1e-8)
    host = wrapper(model, opt)
    for _ in range(2000):
        reset(host)
        step(host, {'members': [(x, y, w)]}, [model], training=True)
    actual = model.log_ratio.detach().exp()
    expected = prior_ratio*p4/p3
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(actual/prior_ratio, p4/p3, rtol=3e-4, atol=3e-5)
    return {'prior_ratio': prior_ratio, 'common_scale': common_scale,
            'learned_odds': actual.tolist(), 'expected_odds': expected.tolist(),
            'max_abs_error': float((actual-expected).abs().max())}


def scale_check():
    x = torch.arange(4).repeat(2)
    y = torch.tensor([0]*4+[1]*4)
    w = torch.tensor([.1,.2,.3,.4,.4,.3,.2,.1])
    outcomes = []
    for scale in (1., 8.):
        model = BinClassifier()
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        host = wrapper(model, opt)
        step(host, {'members': [(x,y,w*scale)]}, [model], training=True)
        outcomes.append((model.log_ratio.detach().clone(), epoch_losses(host, 'train')))
    torch.testing.assert_close(outcomes[1][0], 8*outcomes[0][0], rtol=0, atol=0)
    torch.testing.assert_close(outcomes[1][1], outcomes[0][1], rtol=0, atol=0)
    return {'legacy_gradient_scales_with_global_weights': True,
            'reported_weight_normalized_epoch_metric_invariant': True,
            'note': 'Same optimum under positive global scaling does not mean identical Adam trajectories with epsilon.'}


def signed_counterexample():
    # A negative weight at class 1 makes this empirical objective unbounded
    # below as p(class1) -> 0. This is a scope limit, not a proposed treatment.
    y = torch.tensor([1])
    values = []
    for margin in (-10., -100.):
        logits = torch.tensor([[0., margin]])
        values.append(float((F.cross_entropy(logits,y,reduction='none') * -1.).mean()))
    assert values[1] < values[0] - 80
    return {'loss_at_log_odds_minus_10': values[0], 'loss_at_log_odds_minus_100': values[1],
            'note': 'Positive-weight population odds derivation is not an automatic guarantee for arbitrary signed empirical training weights.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    result = {'known_ratio': [learn(1.,1.), learn(3.,1.), learn(1.,8.)],
              'scaling': scale_check(), 'signed_scope_limit': signed_counterexample(), 'status': 'PASS'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
