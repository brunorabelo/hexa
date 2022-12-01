import torch.nn as nn

import pytorch_lightning as pl
import torch.nn as nn
def linear_model():
    h = 128

    # mlp = nn.Sequential(nn.Linear(13, h), nn.Dropout(0.1), nn.ReLU(),
    #                     nn.BatchNorm1d(h), nn.Linear(h, h), nn.Dropout(0.1), nn.ReLU(),
    #                     nn.BatchNorm1d(h), nn.Linear(h, h), nn.Dropout(0.1), nn.ReLU(),
    #                     nn.BatchNorm1d(h), nn.Linear(h, h), nn.Dropout(0.1), nn.ReLU(),
    #                     nn.Linear(h, 1))

    # mlp = nn.Sequential(nn.Linear(13, h), nn.ReLU(),
    #                     nn.Linear(h, h), nn.ReLU(),
    #                     nn.Linear(h, h), nn.ReLU(),
    #                     nn.Linear(h, h), nn.ReLU(),
    #                     nn.Linear(h, 1))

    mlp = nn.Sequential(nn.Linear(13, h), nn.ReLU(),
                        nn.Linear(h, h), nn.Dropout(0.2), nn.ReLU(),
                        nn.Linear(h, 1))

    return mlp


# define the LightningModule
class MLP(pl.LightningModule):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        z = self.mlp(x)
        loss = nn.functional.l1_loss(y, z)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        z = self.mlp(x)
        loss = nn.functional.l1_loss(y, z)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters())
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    