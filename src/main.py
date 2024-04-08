import torch
import numpy as np
from src.earlystopping import EarlyStopping
from src.data_process import dataset_train,dataset_test,dataset_val,inputs_size,scaler_input,scaler_label
from torch.utils.data import DataLoader
from src.model import Model
from plot import prediction_label_view,loss_view
from train import train
from validation import validation
from utility import adjust_lr
import time
import os
import argparse
import wandb
<<<<<<< HEAD



# test
def test(model,criterion,test_loader):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for _,data in enumerate(test_loader):
            inputs ,target = data
            output = model(inputs)

            #anti-normalization
            output =torch.tensor(scaler_label.inverse_transform(output)).float()
            target = torch.tensor(scaler_label.inverse_transform(target)).float()

            loss = criterion(output, target)
            running_loss += loss.item()
=======



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
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
        running_loss = running_loss/len(test_loader)
    return running_loss


<<<<<<< HEAD


def train_val(timestamp,adjust_lr,inputs_size):

   #prepare the model,criterion,optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    model = Model(inputs_size=inputs_size, hidden_size=wandb.config["hidden_size"])
    optimizer = torch.optim.SGD(model.parameters(), wandb.config["lr"], nesterov=True, momentum=0.9)

    # load data
    train_loader = DataLoader(dataset=dataset_train, batch_size=wandb.config["batch_size"], shuffle=True)
    validation_loader = DataLoader(dataset=dataset_val, batch_size=wandb.config["batch_size"], shuffle=False)

    #start training and validation, record the loss
=======
def train_val(timestamp,adjust_lr,inputs_size):
    #hyperparameters
    global val_loss_aray
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
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
    early_stopping = EarlyStopping(patience=5)
    running_loss_array = np.array([])
    val_loss_array = np.array([])
    best_loss = 100.0
<<<<<<< HEAD


    for epoch in range(1,wandb.config["epoch"]+1):

        adjust_lr(epoch,optimizer,wandb.config["lr"])
        running_loss= train(model,criterion,optimizer,train_loader)
        wandb.log({"epoch":epoch,"running_loss":running_loss})

        val_loss= validation(model,criterion,validation_loader)
=======
    wandb.watch(model, log="all")
    for epoch in range(1,epoch+1):
        adjust_lr(epoch,optimizer,lr)
        running_loss= train(model,criterion,optimizer,train_loader)
        wandb.log({"epoch":epoch,"running_loss":running_loss})
        val_loss, val_loss_aray= validation(model,criterion,validation_loader)
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
        wandb.log({"epoch":epoch,"validation loss":val_loss})
        running_loss_array= np.append(running_loss_array, running_loss)
        val_loss_array= np.append(val_loss_array, val_loss)

        early_stopping(val_loss, model)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'../checkpoint/{timestamp}/checkpoint.pt')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'epoch={epoch} , running loss={running_loss : .4f}')
<<<<<<< HEAD
        if epoch % 2 == 0:
            print(f'validation loss:{val_loss}')

    return running_loss_array,val_loss_array,model

def main():
    # train n times and get n models
    for ep in range(args.experiment_number):

        criterion = torch.nn.MSELoss(reduction='mean')

        # create the files with timestamp
        time_now = time.time()
        local_time = time.localtime(time_now)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        file_name = f"../checkpoint/{timestamp}"
        os.makedirs(file_name)

        # load test data
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)

        #initialize the wandb project,start training and validation
        wandb.init(project="MLP-3",config=args, name= f"{timestamp}",reinit=True)

        #train model and test it
        print(f"The {local_time.tm_hour}clock {local_time.tm_min}minute {local_time.tm_sec}second model's trainning process ")
        running_loss_array,val_loss_aray,model= train_val(timestamp,adjust_lr,inputs_size)
        test_loss = test(model,criterion,test_loader)
        wandb.log({"test_loss":test_loss})
        print(f"this model's test loss is :{test_loss} ")

        # draw and save the images in each file
        loss_view(running_loss_array, val_loss_aray,timestamp)
        prediction_label_view(model, test_loader,timestamp)

    wandb.finish()


if __name__ == '__main__':
    #hyperparameters
=======
    return running_loss_array,val_loss_aray,model

def main():
    # train n times and get n models
    for ep in range(args.experiment_number):
        # create the files with timestamp
        time_now = time.time()
        local_time = time.localtime(time_now)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        file_name = f"../checkpoint/{timestamp}"
        os.makedirs(file_name)

        #initialize the wandb project,start training and validation
        wandb.init(project="MLP",name= f"{timestamp}",reinit=True)

        print(f"The{local_time.tm_hour}hour {local_time.tm_min}minute{local_time.tm_sec}second model's trainning process ")
        running_loss_array,val_loss_aray,model= train_val(timestamp,adjust_lr,inputs_size)

        # draw and save the images in each file
        loss_view(running_loss_array, val_loss_aray,timestamp)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)
        prediction_label_view(model, test_loader,timestamp)





if __name__ == '__main__':
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
    parser = argparse.ArgumentParser(description="about hyperparameter")
    parser.add_argument("--epoch",type=int,default=100,help="epoch")
    parser.add_argument("--lr", type=float,default=0.1, help="learning rate")
    parser.add_argument("--hidden_size", type=int,default=20, help="hidden size")
    parser.add_argument("--batch_size", type=int,default=170, help="batch size")
<<<<<<< HEAD
    parser.add_argument("--experiment_number", type=int,default=10, help="experiment_number")
    args = parser.parse_args()


    isTrain = input("Do you want Train model (True) or Test model(False):")
    if isTrain:
        main()
    else:
        #test
        test_loader = DataLoader(dataset=dataset_test, batch_size=wandb.config["batch_size"], shuffle=False)
=======
    parser.add_argument("-n","--experiment_number", type=int,default=10, help="experiment_number")
    args = parser.parse_args()

    Flag = input("train model (True) or test model(False)")
    if Flag:
        wandb.init(project="MLP")
        main()
        wandb.finish()
    else:
        #test
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
        criterion = torch.nn.MSELoss(reduction='mean')
        model = Model(inputs_size=inputs_size, hidden_size=args.hidden_size)
        timestamp =input("type the name of model you want to test")
        model = model.load_state_dict(torch.load(f'../checkpoint/{timestamp}/checkpoint.pt'))
<<<<<<< HEAD
        running_loss = test(model,criterion,test_loader)
        print(f"the testing loss is {running_loss}")
=======
        running_loss = test(model,criterion,batch_size=args.batch_size)
        print(f"the testing loss is {running_loss}")
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
