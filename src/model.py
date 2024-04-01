import torch
<<<<<<< HEAD


class Model(torch.nn.Module):
    def __init__(self,inputs_size,hidden_size):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(inputs_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, 1)
=======
from sklearn.neural_network import MLPRegressor


class Model(torch.nn.Module):
    def __init__(self,inputs_size):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(inputs_size, 20)
        self.linear2 = torch.nn.Linear(20, 20)
        self.linear3 = torch.nn.Linear(20, 1)
>>>>>>> efc809b30616d9a2f2ba5a4b88ce989dd4eee2b0
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

