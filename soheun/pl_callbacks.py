from pytorch_lightning.callbacks import Callback
from events_data import EventsData
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from plots import calibration_plot, hist_events_by_labels, plot_reweighted_samples


class FvTScorePlotCallback(Callback):
    def __init__(self, events_plot: EventsData, **kwargs):
        super().__init__()
        self.events_plot = events_plot
        self.plot_kwargs = kwargs

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.eval()
        fvt_scores = (
            pl_module.predict(self.events_plot.X_torch).detach().cpu().numpy()[:, 1]
        )

        fig, ax = plt.subplots(1, 1, figsize=self.plot_kwargs.get("figsize", (8, 6)))
        bins = self.plot_kwargs.get("bins", np.linspace(0, 1, 30))
        hist_events_by_labels(self.events_plot, fvt_scores, bins=bins, ax=ax)
        plt.show()
        plt.close("all")

        pl_module.train()


class CalibrationPlotCallback(Callback):
    def __init__(self, events_plot: EventsData, **kwargs):
        super().__init__()
        self.events_plot = events_plot
        self.plot_kwargs = kwargs
        self.title = kwargs.get("title", "")
        self.plot_every_n_epochs = kwargs.get("plot_every_n_epochs", 1)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return

        pl_module.eval()
        print("pl_module.device", pl_module.device)
        pl_module.to("cuda")
        fvt_scores = (
            pl_module.predict(self.events_plot.X_torch).detach().cpu().numpy()[:, 1]
        )
        fig, ax = plt.subplots(1, 1, figsize=self.plot_kwargs.get("figsize", (8, 6)))
        fig.suptitle(f"Epoch {trainer.current_epoch}, {self.title}")
        bins = self.plot_kwargs.get("bins", 30)
        calibration_plot(
            fvt_scores,
            self.events_plot.is_4b,
            bins=bins,
            ax=ax,
            sample_weights=self.events_plot.weights,
        )
        plt.show()
        plt.close("all")

        pl_module.train()


class ReweightedPlotCallback(Callback):
    def __init__(self, events_plot: EventsData, **kwargs):
        super().__init__()
        self.events_plot = events_plot
        self.plot_kwargs = kwargs
        self.title = kwargs.get("title", "")
        self.plot_every_n_epochs = kwargs.get("plot_every_n_epochs", 1)
        self.ratio_4b = kwargs.get("ratio_4b", 0.5)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return

        pl_module.eval()
        print("pl_module.device", pl_module.device)
        pl_module.to("cuda")
        fvt_scores = (
            pl_module.predict(self.events_plot.X_torch).detach().cpu().numpy()[:, 1]
        )
        reweights = (
            (fvt_scores / (1 - fvt_scores)) * self.ratio_4b / (1 - self.ratio_4b)
        )
        fig, ax = plt.subplots(1, 1, figsize=self.plot_kwargs.get("figsize", (8, 6)))
        fig.suptitle(f"Epoch {trainer.current_epoch}, {self.title}")
        bins = self.plot_kwargs.get("bins", 30)
        plot_reweighted_samples(
            self.events_plot,
            fvt_scores,
            reweights,
            ax=ax,
            bins=bins,
            mode="uniform",
        )
        plt.show()
        plt.close("all")

        pl_module.train()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.eval()
        print("pl_module.device", pl_module.device)
        pl_module.to("cuda")
        fvt_scores = (
            pl_module.predict(self.events_plot.X_torch).detach().cpu().numpy()[:, 1]
        )
        reweights = (
            (fvt_scores / (1 - fvt_scores)) * self.ratio_4b / (1 - self.ratio_4b)
        )
        fig, ax = plt.subplots(1, 1, figsize=self.plot_kwargs.get("figsize", (8, 6)))
        fig.suptitle(f"Epoch {trainer.current_epoch}, {self.title}")
        bins = self.plot_kwargs.get("bins", 30)
        plot_reweighted_samples(
            self.events_plot,
            fvt_scores,
            reweights,
            ax=ax,
            bins=bins,
            mode="uniform",
        )
        plt.show()
        plt.close("all")
