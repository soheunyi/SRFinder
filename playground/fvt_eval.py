# Get gradients of output of clf.net for the validation set
from typing import Literal
import pandas as pd
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from python.classifier.symmetrized_model_train import symmetrizedModelParameters


def ce_loss_with_scores(
    scores: torch.Tensor, y_true: torch.Tensor, w: torch.Tensor, device
):
    return (
        (
            w
            * F.cross_entropy(
                scores,
                y_true,
                weight=torch.FloatTensor([1, 1]).to(device),
                reduction="none",
            )
        )
        .sum(dim=0)
        .to("cpu")
    )


def logit_gradient_with_scores(inputs: torch.Tensor, scores: torch.Tensor):
    logit = scores[:, 1] - scores[:, 0]
    return (
        grad(
            logit,
            inputs,
            create_graph=True,
            retain_graph=False,
            grad_outputs=torch.ones_like(logit),
        )[0]
        .detach()
        .to("cpu")
    )


def calculate_fvt_values(
    clf: symmetrizedModelParameters,
    df: pd.DataFrame,
):
    layer1Pix = "0123"
    required_cols = [f"sym_canJet{i}_pt" for i in layer1Pix]
    required_cols += [f"sym_canJet{i}_eta" for i in layer1Pix]
    required_cols += [f"sym_canJet{i}_phi" for i in layer1Pix]
    required_cols += [f"sym_canJet{i}_m" for i in layer1Pix]
    required_cols += [clf.yTrueLabel]
    required_cols += [clf.weight]

    if not set(required_cols).issubset(set(df.columns)):
        raise ValueError(
            "Missing columns in input dataframe: {}".format(
                set(required_cols) - set(df.columns)
            )
        )

    J_val, y_val, w_val = clf.dfToTensors(df, y_true=clf.yTrueLabel)

    val_evalLoader = DataLoader(
        dataset=TensorDataset(J_val, y_val, w_val),
        batch_size=2**14,
        shuffle=False,
        pin_memory=True,
    )

    loss = 0
    w_sum = 0
    grads = torch.tensor([])
    scores = torch.tensor([])

    clf.net.eval()

    for J, y, w in val_evalLoader:
        J = J.requires_grad_().to(clf.net.device)
        y = y.to(clf.net.device)
        w = w.to(clf.net.device)

        c_score, _ = clf.net(J)

        loss += ce_loss_with_scores(c_score, y, w, clf.net.device)
        w_sum += w.sum().cpu().item()

        grad_ = logit_gradient_with_scores(J, c_score)
        grads = torch.cat((grads, grad_))

        scores = torch.cat((scores, c_score.detach().to("cpu")))

    loss = loss.item() / w_sum

    return loss, grads, scores
