from pytorch_lightning.callbacks import Callback
from events_data import EventsData
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from plots import hist_events_by_labels


class FvTScorePlotCallback(Callback):
    def __init__(self, events_plot: EventsData, **kwargs):
        super().__init__()
        self.events_plot = events_plot
        self.plot_kwargs = kwargs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.eval()
        self.events_plot.set_model_scores(pl_module)

        fig, ax = plt.subplots(
            1, 1, figsize=self.plot_kwargs.get("figsize", (8, 6)))
        bins = self.plot_kwargs.get("bins", np.linspace(0, 1, 30))
        hist_events_by_labels(
            self.events_plot, self.events_plot.fvt_score, bins=bins, ax=ax
        )
        plt.show()
        plt.close("all")

        pl_module.train()
