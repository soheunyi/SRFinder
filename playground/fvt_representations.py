import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def get_fvt_reprs(X, model, device=torch.device("cuda:0")):
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    model = model.to(device)
    q_repr = []
    view_scores = []

    for batch in loader:
        x = batch[0].to(device)
        q = model.encoder(x)
        q_repr.append(q.detach().cpu().numpy())

        view_scores_batch = model.select_q(q)
        view_scores_batch = F.softmax(view_scores_batch, dim=-1)
        view_scores.append(view_scores_batch.detach().cpu().numpy().reshape(-1, 3))

    q_repr = np.concatenate(q_repr, axis=0)
    view_scores = np.concatenate(view_scores, axis=0)

    return q_repr, view_scores
