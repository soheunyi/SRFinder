from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from datetime import datetime
import logging


class FileHandlerLogger(Logger):
    def __init__(self, file_handler: logging.FileHandler):
        super().__init__()
        self.file_handler = file_handler

    @property
    def name(self):
        return "FileHandlerLogger"

    @property
    def version(self):
        return "1.0"

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for k, v in metrics.items():
            self.file_handler.stream.write(f"[{current_time}] {k}: {v}\n")
        self.file_handler.stream.flush()

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    def save(self):
        pass
