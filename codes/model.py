import torch
from sklearn.neural_network import MLPRegressor
from codes.dataset import inputs_size,X_train,y_train
from sklearn.model_selection import GridSearchCV


class Model(torch.nn.Module):
    def __init__(self,inputs_size=11):
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

param_grid ={
    "activation":["relu","logistic"],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    "hidden_layer_sizes":[(20),(20,20),(20,20,20),(20,20,20,20),(20,20,20,20,20)],
    "batch_size":[341,170,85,40,20]
}
if __name__ == '__main__':
    model = MLPRegressor(Model)
    grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5)
    grid_search.fit(X_train,y_train.ravel())
    print(f'最佳参数为：{grid_search.best_params_}')
    print(f'最佳参数的得分为：{grid_search.best_score_}')
