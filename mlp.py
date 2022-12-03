import torch.nn as nn
import pytorch_lightning as pl
import torch.nn as nn
import torch

# define the LightningModule
class MLP(pl.LightningModule):
    def __init__(self, h, d):
        super().__init__()
        if d > 0:
            self.mlp = nn.Sequential(nn.Linear(13, h), nn.ReLU(),
                            nn.Linear(h, h), nn.Dropout(d), nn.ReLU(),
                            nn.Linear(h, 1))
        else:
            self.mlp = nn.Sequential(nn.Linear(13, h), nn.ReLU(),
                            nn.Linear(h, h), nn.ReLU(),
                            nn.Linear(h, h), nn.ReLU(),
                            nn.Linear(h, h), nn.ReLU(),
                            nn.Linear(h, 1))

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

    # def configure_optimizers(self):
    #     # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
    #     # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        opt = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

        return opt
    