from pytorch_lightning.callbacks import Callback
from events_data import EventsData
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import torch

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


class SaveIndividualClassifierCallback(pl.Callback):
    def __init__(
        self,
        save_dir: str,
        run_names: list[str],
        monitor_metrics: list[str],
        model="FvTClassifier",
    ):
        assert model in ["FvTClassifier", "AttentionClassifier"]
        super().__init__()
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        assert len(run_names) == len(monitor_metrics)
        self.run_names = run_names
        self.monitor_metrics = monitor_metrics
        self.best_scores = {metric: float("inf") for metric in monitor_metrics}
        self.model = model

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        for i, (run_name, metric) in enumerate(
            zip(self.run_names, self.monitor_metrics)
        ):
            current_score = trainer.callback_metrics.get(metric, None)
            if current_score is None:
                continue

            # Convert from tensor if needed
            if hasattr(current_score, "item"):
                current_score = current_score.item()

            # Save if better than previous best
            if current_score < self.best_scores[metric]:
                self.best_scores[metric] = current_score
                save_path = self.save_dir / f"{run_name}_best.pt"
                if self.model == "FvTClassifier":
                    torch.save(pl_module.fvt_classifiers[i].state_dict(), save_path)
                elif self.model == "AttentionClassifier":
                    torch.save(
                        pl_module.attention_classifiers[i].state_dict(), save_path
                    )
                else:
                    raise ValueError(f"Invalid model: {self.model}")

            # Always save latest
            save_path = self.save_dir / f"{run_name}_last.pt"
            if self.model == "FvTClassifier":
                torch.save(pl_module.fvt_classifiers[i].state_dict(), save_path)
            elif self.model == "AttentionClassifier":
                torch.save(pl_module.attention_classifiers[i].state_dict(), save_path)
            else:
                raise ValueError(f"Invalid model: {self.model}")
