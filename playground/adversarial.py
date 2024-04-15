from matplotlib import pyplot as plt
from classifier import MLPClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AffineAdversarialNet(nn.Module):
    def __init__(self, classifier: MLPClassifier):
        super().__init__()
        self.classifier = classifier
        self.transform = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.classifier.forward(x)

    def predict(self, x: torch.Tensor):
        return self.classifier.predict(x)

    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        epochs=200,
        verbose=False,
        test_split=0.2,
        save_epochs=0,
    ):
        """
        X: inputs to transform
        Y: inputs not to transform
        save_epochs: save model every save_epochs epochs. 0 to disable
        """

        n_X_samples = X.shape[0]
        n_Y_samples = Y.shape[0]

        train_covariates = torch.cat([X, Y])
        train_labels = torch.cat([torch.zeros(n_X_samples), torch.ones(n_Y_samples)])

        criterion = nn.BCELoss()

        train_loader = DataLoader(
            TensorDataset(train_covariates, train_labels), batch_size=32, shuffle=True
        )

        optimizer_transform = torch.optim.Adam(self.transform.parameters(), lr=1e-4)
        optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)

        train_losses_record = []

        for epoch in range(epochs):
            for it, (cov_batch, labels_batch) in enumerate(train_loader):
                optimizer_transform.zero_grad()
                optimizer_classifier.zero_grad()

                transformed = cov_batch
                transformed[labels_batch == 0] = self.transform(
                    cov_batch[labels_batch == 0]
                )

                out = self.classifier(transformed)

                step_classifier = it % 2

                if step_classifier:
                    loss = criterion(out, labels_batch.reshape(-1, 1))
                    loss.backward()
                    optimizer_classifier.step()

                else:
                    loss = -criterion(out, labels_batch.reshape(-1, 1))
                    loss.backward()
                    optimizer_transform.step()

            train_losses_record.append(loss.item() * (1 if step_classifier else -1))

            if epoch % 20 == 0 and verbose:
                print("Epoch: {}, Train loss: {}".format(epoch, loss.item()))

            if epoch % save_epochs == 0:
                torch.save(self.state_dict(), f"models/adversarial_epoch_{epoch}.pt")

        if verbose:
            plt.plot(train_losses_record, label="train")
            plt.legend()
            plt.show()

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
