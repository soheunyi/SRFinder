from torch.utils.data import DataLoader
import pytorch_lightning as pl


class FvTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        num_workers=4,
        batch_size_milestones=[],
        batch_size_multiplier=2,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_size_multiplier = batch_size_multiplier
        self.batch_size_milestones = batch_size_milestones

    def train_dataloader(self):
        if self.trainer.current_epoch in self.batch_size_milestones:
            self.batch_size *= self.batch_size_multiplier
            print(f"Batch size updated to: {self.batch_size}")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
