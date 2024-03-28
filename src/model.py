import torch
from sklearn.neural_network import MLPRegressor


class Model(torch.nn.Module):
    def __init__(self,inputs_size):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(inputs_size, 20)
        self.linear2 = torch.nn.Linear(20, 20)
        self.linear3 = torch.nn.Linear(20, 1)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

