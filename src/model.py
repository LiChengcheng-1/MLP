import torch
<<<<<<< HEAD
=======
<<<<<<< HEAD


class Model(torch.nn.Module):
    def __init__(self,inputs_size,hidden_size):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(inputs_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, 1)
=======
from sklearn.neural_network import MLPRegressor
>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398


class Model(torch.nn.Module):
    def __init__(self,inputs_size,hidden_size):
        super(Model,self).__init__()
<<<<<<< HEAD
        self.linear1 = torch.nn.Linear(inputs_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear4 = torch.nn.Linear(hidden_size, 1)
=======
        self.linear1 = torch.nn.Linear(inputs_size, 20)
        self.linear2 = torch.nn.Linear(20, 20)
        self.linear3 = torch.nn.Linear(20, 1)
>>>>>>> efc809b30616d9a2f2ba5a4b88ce989dd4eee2b0
>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

