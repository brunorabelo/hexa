from torch.utils.data import Dataset
import torch
import numpy as np

class TweetsDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, dataset):
        """
        Args:
            data: pd dataframe
        """
        self.dataset = dataset
        self.target = "retweets_count"

        self.X = dataset.drop(self.target, axis=1)
        self.y = dataset[self.target]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values.astype(np.float32), np.array([self.y[idx].astype(np.float32)])]

class TweetsDatasetEmbedded(Dataset):
    """Students Performance dataset."""

    def __init__(self, dataset, embed_len):
        """
        Args:
            data: pd dataframe
        """
        self.embed_len = embed_len
        self.dataset = dataset
        self.target = "retweets_count"
        self.no_target = False

        if self.target in dataset.columns:
            self.X = dataset.drop(self.target, axis=1)
            self.y = dataset[self.target]
        else:
            print("Target column not present")
            self.X = dataset
            self.no_target = True
        
        data = self.X.drop([str(i) for i in range(self.embed_len)], axis=1)
        self.data = data.values.astype(np.float32)
        text = self.X[[str(i) for i in range(self.embed_len)]]
        self.text = np.expand_dims(text.values.astype(np.float32), axis=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        
        # return [text.astype(np.float32), data.values.astype(np.float32), np.array([self.y[idx].astype(np.float32)])]
        if self.no_target:
            return [self.text[idx], self.data[idx]]
        
        return [self.text[idx], self.data[idx], np.array([self.y[idx].astype(np.float32)])]
