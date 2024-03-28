import torch
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    def __init__(self,inputs,label):

        self.x_data = torch.from_numpy(inputs).float()
        self.y_data = torch.from_numpy(label).float()

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return len(self.x_data)

