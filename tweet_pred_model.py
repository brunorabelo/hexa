
import pytorch_lightning as pl
import torch.nn as nn

import torch
class TweetPredModel(pl.LightningModule):

    def __init__(self, embed_len, data_len, output_channel, dense_size):
        super().__init__()
        """
        n_dim: embedding dimensions
        """
        self.embed_len = embed_len
        self.data_len = data_len
        self.conv1 = nn.Conv1d(embed_len, 16, kernel_size=4, padding="same")
        self.linear1 = nn.Linear(16, 16)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=4, padding="same")
        self.linear2 = nn.Linear(8, output_channel)

        self.linear3 = nn.Linear(output_channel+data_len, dense_size)
        self.linear4 = nn.Linear(dense_size, dense_size)
        self.linear5 = nn.Linear(dense_size, 1)


    def forward(self, text, data):
        # x1 = self.conv1(text)
        # x1 = self.linear1(x1)
        # x1 = nn.ReLU(x1)
        x1 = self.conv2(text)
        x1 = self.linear2(x1)
        x1 = nn.ReLU(x1)
        

        x2 = data
        x = torch.cat((x1, x2), dim=2)
        x = self.linear3(x)
        x = nn.ReLU(x)
        x = self.linear4(x)
        x = nn.ReLU(x)
        x = self.linear5(x)

    def training_step(self, batch, batch_idx):
        text, data, y = batch
        print(text,data)
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
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer