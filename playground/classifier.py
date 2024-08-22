from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from torch.func import hessian


class MLPClassifier(nn.Module):
    # Input: 2d
    # Output: probability of class 4b (1d array)
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        out = F.sigmoid(self.fc6(x))
        return out

    def predict(self, x):
        return self.forward(x)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        run_name,
        epochs=200,
        verbose=False,
        use_gpu=True,
    ):
        # X: 2d array with shape (n_samples, xy)
        # y: 1d array with shape (n_samples, labels)

        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        print("Using device:", device)

        criterion = nn.BCELoss()

        X_train, y_train = train_loader.dataset.tensors
        X_valid, y_valid = valid_loader.dataset.tensors

        # send data to device
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_valid, y_valid = X_valid.to(device), y_valid.to(device)

        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        best_valid_loss = torch.inf

        train_losses_record = []
        valid_losses_record = []

        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch.reshape(-1, 1))
                loss.backward()
                optimizer.step()

            train_criterion = nn.BCELoss(reduction="mean")
            with torch.no_grad():
                outputs = self.forward(X_train)
                train_loss = train_criterion(outputs, y_train.reshape(-1, 1))
                train_losses_record.append(train_loss.to("cpu"))

            valid_criterion = nn.BCELoss(reduction="mean")
            with torch.no_grad():
                outputs = self.forward(X_valid)
                valid_loss = valid_criterion(outputs, y_valid.reshape(-1, 1))
                valid_losses_record.append(valid_loss.to("cpu"))

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.state_dict(), f"models/best_{run_name}.pt")

                if verbose and epoch % 10 == 0:
                    print(
                        "Epoch: {}, Train Loss: {}, Valid Loss: {}".format(
                            epoch, train_loss, valid_loss
                        )
                    )

        if verbose:
            plt.plot(train_losses_record, label="train")
            plt.plot(valid_losses_record, label="valid")
            plt.legend()
            plt.show()

    def get_input_gradients(self, X: torch.Tensor):
        return get_gradient(self.forward, X)

    def get_input_hessian(self, X: torch.Tensor):
        return get_hessian(self.forward, X)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


def get_hessian(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor):
    X.requires_grad = True

    return torch.stack([hessian(f)(x) for x in X])


def get_gradient(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor):
    X = X.clone().detach().requires_grad_(True)
    y = f(X)
    y.backward(torch.ones_like(y))
    return X.grad


def get_laplacian(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor):
    X.requires_grad = True
    y = f(X)
    d1y = grad(y, X, create_graph=True, grad_outputs=torch.ones_like(y))[0]
    d2y = grad(d1y, X, grad_outputs=torch.ones_like(d1y))[0]
    return torch.sum(d2y, dim=-1)
