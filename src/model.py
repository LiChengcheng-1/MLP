import torch

# class Model(torch.nn.Module):
#     def __init__(self,inputs_size,hidden_size):
#         super(Model,self).__init__()
#         self.linear1 = torch.nn.Linear(inputs_size, hidden_size)
#         self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
#         self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
#         self.linear4 = torch.nn.Linear(hidden_size, 1)
#         self.relu = torch.nn.ReLU()
#     def forward(self,x):
#         x = self.relu(self.linear1(x))
#         x = self.relu(self.linear2(x))
#         x = self.relu(self.linear3(x))
#         x = self.linear4(x)
#         return x

class Model(torch.nn.Module):
    def __init__(self, inputs_size, hidden_size,hidden_layers):
        super(Model, self).__init__()
        self.layers = hidden_layers
        self.input_layer = torch.nn.Linear(inputs_size, hidden_size)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(self.layers)])
        self.output_layer = torch.nn.Linear(hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for i in range(self.layers):
            x=self.relu(self.linears[i](x))
        x = self.output_layer(x)
        return x
