import numpy as np
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback
import torch
from sklearn.decomposition import PCA
import rtd


class RandomModel(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_channels, out_channels)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        return out  # - torch.mean(out)) / torch.std(out)


class OneToRandomRTDCallback(Callback):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        device: torch.device = torch.device("cpu"),
        rtd_batch_size: int = 500,
        rtd_n_trials: int = 10,
        pca: bool = False,
    ) -> None:
        super().__init__()

        seed = torch.seed()
        torch.manual_seed(0)
        self.random_model = RandomModel(in_channels, out_channels).to(device)
        torch.manual_seed(seed)

        self.device = device
        self.rtd_batch_size = rtd_batch_size
        self.rtd_n_trials = rtd_n_trials
        self.pca = pca

    # @staticmethod
    # def normalize(x: np.array) -> np.array:
    #     return (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        random_embeddings = []
        embeddings = []

        pl_module.eval()
        with torch.no_grad():
            for batch in trainer.datamodule.val_dataloader():
                img, _ = batch
                img = img.to(self.device)
                random_embeddings.append(self.random_model(img.flatten(1)))
                embeddings.append(pl_module.get_embeddings(img))

        random_embeddings = torch.concat(random_embeddings, dim=0).cpu().numpy()
        embeddings = torch.concat(embeddings, dim=0).cpu().numpy()
        if self.pca:
            pca = PCA(n_components=2).fit(embeddings)
            embeddings = pca.transform(embeddings)

        pl_module.log("RTD_score", rtd.rtd(random_embeddings, embeddings, trials=self.rtd_n_trials, batch=self.rtd_batch_size))
