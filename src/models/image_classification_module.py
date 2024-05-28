from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class ImageClassificationModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_params: Dict[str, Any],
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=100)
        self.val_acc = Accuracy(task="multiclass", num_classes=100)
        self.test_acc = Accuracy(task="multiclass", num_classes=100)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.
        """
        return self.net(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        _, embeddings = self.net(x, return_features=True)
        return embeddings

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def log_metrics(self, stage: str) -> None:
        """Log metrics computed during training, validation or testing."""
        on_step = stage == "train"

        self.log(
            f"{stage}/loss",
            getattr(self, f"{stage}_loss").compute(),
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True
        )
        self.log(
            f"{stage}/accuracy",
            getattr(self, f"{stage}_acc").compute(),
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True
        )

        getattr(self, f"{stage}_loss").reset()
        getattr(self, f"{stage}_acc").reset()

    def conduct_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Perform a single model step on a batch of data."""
        loss, predictions, targets = self.model_step(batch)

        getattr(self, f"{stage}_loss").update(loss)
        getattr(self, f"{stage}_acc").update(predictions, targets)

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
        """
        loss = self.conduct_step(batch, "train")
        self.log_metrics("train")

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.
        """
        self.conduct_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc

        self.log_metrics("val")

        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.
        """
        self.conduct_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log_metrics("test")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            scheduler_params = dict(self.hparams.scheduler_params)
            scheduler_params["scheduler"] = scheduler
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_params,
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ImageClassificationModule(None, None, None, None)
