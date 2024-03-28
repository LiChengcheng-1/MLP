import optuna
from model import Model
from src.data_process import inputs_size
import torch.optim as optim
import torch
from src.data_process import dataset_train,dataset_val
from torch.utils.data import DataLoader
from main import train_model,validation



def objective(trial):
  model = Model(inputs_size)
  #try  AdaDelta adn Adagrad
  optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adadelta","Adagrad"])
  optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=0.1)
  batch_size=trial.suggest_int("batch_size", 64, 256,step=64)
  criterion = torch.nn.MSELoss(reduction='mean')


  epoch = 100
  train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
  validation_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

  # Training of the model.
  for epoch in range(1,epoch+1):
    train_model(model,criterion,optimizer,train_loader)
  # Validation of the model.
    val_loss= validation(model,criterion,validation_loader)
    trial.report(val_loss, epoch+1)
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
        return val_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
trial = study.best_trial
print('loss: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))