import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle

class SequenceDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data)  # (num_rows, window_size, num_features)
        self.target = torch.from_numpy(target)  # (num_rows, num_features)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = self.data[i].float()  # (window_size, num_features)
        y = self.target[i].float()

        return x, y


class SequenceDataset_fromfile(Dataset):
    def __init__(self, data_path="./data", target_path="./target'"):
        with open(data_path, 'rb') as f:
            self.data = torch.tensor(pickle.load(f))
        with open(target_path, 'rb') as f:
            self.target = torch.tensor(pickle.load(f))  # (num_rows, num_features)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = self.data[i].float()  # (window_size, num_features)
        y = self.target[i].float()

        return x, y


class SequenceDataset_from_dataframe(Dataset):
    def __init__(self, data, window_size):
        self.dataframe = data
        self.window_size = window_size

    def __len__(self):
        return len(self.dataframe) - self.window_size

    def __getitem__(self, i):
        # Get input and target from dataframe
        x = self.dataframe.iloc[i : i + self.window_size].values
        y = self.dataframe.iloc[i + self.window_size].values
        # To tensor
        x = torch.tensor(x).float()  # (window_size, feature_num)
        y = torch.tensor(y).float()  # (, feature_num)

        return x, y


class SequenceDataset_from_dataframe_multitask(Dataset):
    def __init__(self, data, category_features, non_category_features, window_size):
        self.dataframe1 = data[category_features]
        self.dataframe2 = data[non_category_features]
        self.window_size = window_size

    def __len__(self):
        return len(self.dataframe1) - self.window_size

    def __getitem__(self, i):
        # Get input and target from dataframe
        category_input = self.dataframe1.iloc[i: i + self.window_size].values
        category_target = self.dataframe1.iloc[i + self.window_size].values
        noncategory_input = self.dataframe2.iloc[i: i + self.window_size].values
        noncategory_target = self.dataframe2.iloc[i + self.window_size].values
        # To tensor
        category_input = torch.tensor(category_input).float()  # (window_size, feature_num)
        noncategory_input = torch.tensor(noncategory_input).float()  # (window_size, feature_num)
        whole_input = torch.cat((category_input, noncategory_input), dim=1)
        category_target = torch.tensor(category_target).float()  # (, feature_num)
        noncategory_target = torch.tensor(noncategory_target).float()  # (, feature_num)

        return whole_input, (category_target, noncategory_target)