import torch
from model import Model
from data_process import inputs_size,dataset_train
import wandb
from torch.utils.data import DataLoader
from train import train



# using sweep to choose hyperparameter
if __name__ == '__main__':
    # Pass defaults to wandb.init
    sweep_config = {
        'method': 'grid'
    }
    metric = {
        'name': 'running_loss',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'learning_rate': {
            'values': [0.001, 0.01, 0.1]
        },
        'batch_size': {
            'values': [60, 170, 340]
        },
        'hidden_size': {
            'values': [20, 30, 40]
        },
        'epoch': {
            'values': [10, 50, 100]
        },
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="MLP")


    def sweep(config=None):
        with wandb.init(config=config):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
            criterion = torch.nn.MSELoss(reduction='mean')
            model = Model(inputs_size=inputs_size, hidden_size=config.hidden_size)
            optimizer = torch.optim.SGD(model.parameters(), config.learning_rate, nesterov=True, momentum=0.9)
            train_loader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
            #train
            for epoch in range(1, config.epoch + 1):
                running_loss = train(model, criterion, optimizer, train_loader)
                wandb.log({"epoch": epoch, "running_loss": running_loss})

    wandb.agent(sweep_id, sweep)