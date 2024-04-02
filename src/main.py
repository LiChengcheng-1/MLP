import torch
import numpy as np
from src.earlystopping import EarlyStopping
from src.data_process import dataset_train,dataset_test,dataset_val,inputs_size,scaler_input,scaler_label
from torch.utils.data import DataLoader
from src.model import Model
<<<<<<< HEAD
from plot import prediction_label_view,loss_view
=======
<<<<<<< HEAD
from plot import prediction_label_view,loss_view
from train import train
from validation import validation
import time
import os
import argparse



#test
def test(model,criterion,batch_size):
    model.eval()
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    running_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            batch_loss = criterion(output, target)
            running_loss += batch_loss.item()
        running_loss = running_loss/len(test_loader)
    return running_loss


def adjust_lr(epoch,optimizer,lr):
    # optimize lr
    step = [10, 20, 30, 40]
    base_lr = lr
=======
from plot import prediction_label_view
>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
from train import train
from validation import validation
import time
import os
import argparse
import wandb



#test
def test(model,criterion,batch_size):
    model.eval()
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    running_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = scaler_input.inverse_transform(data)
            target =scaler_label.inverse_transform(target)
            output = model(data)
            batch_loss = criterion(output, target)
            running_loss += batch_loss.item()
        running_loss = running_loss/len(test_loader)
    return running_loss


def adjust_lr(epoch,optimizer,lr):
    # optimize lr
    step = [10, 20, 30, 40]
<<<<<<< HEAD
    base_lr = lr
=======
    base_lr = 0.1

>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
    lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr
    return lr


<<<<<<< HEAD
def train_val(timestamp,adjust_lr,inputs_size):
=======
<<<<<<< HEAD
def train_val(timestamp,adjust_lr,inputs_size):
    #hyperparameters
    epoch = args.epoch
    inputs_size = inputs_size
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    lr=args.lr

   #prepare the model,criterion,optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    model = Model(inputs_size=inputs_size, hidden_size=hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr, nesterov=True, momentum=0.9)

    # load data
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=5)
    running_loss_array = np.array([])
    best_loss = 100.0
    for epoch in range(1,epoch+1):
        adjust_lr(epoch,optimizer,lr)
        running_loss= train(model,criterion,optimizer,train_loader)
        val_loss, val_loss_aray= validation(model,criterion,validation_loader)
=======
def main(num_model,model,adjust_lr):
>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
    #hyperparameters
    epoch = args.epoch
    inputs_size = inputs_size
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    lr=args.lr
    # save the hypterparameters in wandb
    config = wandb.config
    config.epoch = args.epoch
    config.inputs_size = inputs_size
    config.hidden_size = args.hidden_size
    config.batch_size = args.batch_size
    config.lr = args.lr
   #prepare the model,criterion,optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    model = Model(inputs_size=inputs_size, hidden_size=hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr, nesterov=True, momentum=0.9)

    # load data
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    #start training and validation
    early_stopping = EarlyStopping(patience=5)
    running_loss_array = np.array([])
    best_loss = 100.0
    wandb.watch(model, log="all")
    for epoch in range(1,epoch+1):
        adjust_lr(epoch,optimizer,lr)
        running_loss= train(model,criterion,optimizer,train_loader)
        wandb.log({"epoch":epoch,"running_loss":running_loss})
        val_loss, val_loss_aray= validation(model,criterion,validation_loader)
<<<<<<< HEAD
        wandb.log({"epoch":epoch,"validation loss":val_loss})
=======


>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
        running_loss_array= np.append(running_loss_array, running_loss)
        early_stopping(val_loss, model)
        if val_loss < best_loss:
            best_loss = val_loss
<<<<<<< HEAD
            torch.save(model.state_dict(), f'../checkpoint/{timestamp}/checkpoint.pt')
=======
<<<<<<< HEAD
            torch.save(model.state_dict(), f'../checkpoint/{timestamp}/checkpoint.pt')
=======
            torch.save(model.state_dict(), f'../checkpoint/checkpoint_{num_model}.pt')

>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'epoch={epoch} , running loss={running_loss : .4f}')
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
    return running_loss_array,val_loss_aray,model

def multi_train():
    # train n times and get n models
<<<<<<< HEAD
    for ep in range(args.experiment_number):
        # create the files with timestamp
=======
    for ep in range(args.n):
>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
        time_now = time.time()
        local_time = time.localtime(time_now)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        file_name = f"../checkpoint/{timestamp}"
        os.makedirs(file_name)
<<<<<<< HEAD

        #initialize the wandb project,start training and validation
        wandb.init(project="MLP",name= f"{timestamp}",reinit=True)

        print(f"The{local_time.tm_hour}hour {local_time.tm_min}minute{local_time.tm_sec}second model's trainning process ")
        running_loss_array,val_loss_aray,model= train_val(timestamp,adjust_lr,inputs_size)

=======
        print(f"The {local_time.tm_min} minute model's trainning process ")
        running_loss_array,val_loss_aray,model= train_val(timestamp,adjust_lr,inputs_size)
>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
        # draw and save the images in each file
        loss_view(running_loss_array, val_loss_aray,timestamp)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)
        prediction_label_view(model, test_loader,timestamp)

    # model.load_state_dict(torch.load(f'../checkpoint/checkpoint_{best_model}.pt'))
<<<<<<< HEAD
=======




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="about hyperparameter")
    parser.add_argument("--epoch",type=int,default=100,help="epoch")
    parser.add_argument("--lr", type=float,default=0.1, help="learning rate")
    parser.add_argument("--hidden_size", type=int,default=20, help="hidden size")
    parser.add_argument("--batch_size", type=int,default=170, help="batch size")
    parser.add_argument("-n","--experiment_number", type=int,default=10, help="experiment_number")
    args = parser.parse_args()
    multi_train()
=======
    return running_loss_array,val_loss_aray
>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398




if __name__ == '__main__':
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="about hyperparameter")
    parser.add_argument("--epoch",type=int,default=100,help="epoch")
    parser.add_argument("--lr", type=float,default=0.1, help="learning rate")
    parser.add_argument("--hidden_size", type=int,default=20, help="hidden size")
    parser.add_argument("--batch_size", type=int,default=170, help="batch size")
    parser.add_argument("-n","--experiment_number", type=int,default=10, help="experiment_number")
    args = parser.parse_args()
    wandb.init(project="MLP")
    multi_train()
    wandb.finish()
=======
    multi_train(inputs_size)

>>>>>>> 651d2885f0bd7e7a61ec6e358448b74be3552398
