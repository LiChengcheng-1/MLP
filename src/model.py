import torch


class Model(torch.nn.Module):
    def __init__(self,inputs_size,hidden_size):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(inputs_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, 1)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

