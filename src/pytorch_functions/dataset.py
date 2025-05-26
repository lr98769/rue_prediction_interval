from torch.utils.data import Dataset
import torch


class TabularDataset(Dataset):
    def __init__(self, df, feat_cols, target_cols):
        self.data_df = df
        self.feat_cols = feat_cols
        self.target_cols = target_cols

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        features = torch.from_numpy(row[self.feat_cols].values.astype(float)).float()
        label = torch.from_numpy(row[self.target_cols].values.astype(float)).float()
        return features, label