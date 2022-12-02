
import pytorch_lightning as pl
import torch.nn as nn

import torch
class TweetPredModel(pl.LightningModule):

    def __init__(self, embed_len, data_len, dense_size):
        super().__init__()
        """
        n_dim: embedding dimensions
        """

        self.textConv = nn.Sequential(
            
            # nn.Conv1d(1, 8, kernel_size=3, padding="same"),
            # nn.BatchNorm1d(8),
            # nn.ReLU(),

            # nn.Conv1d(8, 16, kernel_size=3, padding="same"),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),

            nn.Conv1d(1, 4, kernel_size=3, padding="same"),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(),

            nn.Conv1d(4, 16, kernel_size=3, padding="same"),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(),

            nn.Flatten()
        )

        # self.linear3 = nn.Linear(embed_len*16 + data_len, dense_size)
        self.linear3 = nn.Linear(95, dense_size)
        self.linear4 = nn.Linear(dense_size, dense_size)
        self.linear5 = nn.Linear(dense_size, 1)


    def forward(self, text, data):
        # print(text.shape)
        x1 = self.textConv(text)
        # print(x1.shape)
        x2 = data
        # print(x2.shape)
        x = torch.cat((x1, x2), dim=1)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.linear4(x)
        x = nn.ReLU()(x)
        x = self.linear5(x)

        return x


    def training_step(self, batch, batch_idx):
        text, data, y = batch
        # data = data.view(x.size(0), -1)
        z = self.forward(text, data)
        loss = nn.functional.l1_loss(y, z)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        text, data, y = batch
        # data = data.view(x.size(0), -1)
        z = self.forward(text, data)
        loss = nn.functional.l1_loss(y, z)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters())
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        return [optimizer], [scheduler]