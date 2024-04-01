import torch
import numpy as np
from src.earlystopping import EarlyStopping
from src.data_process import dataset_train,dataset_test,dataset_val,inputs_size
from torch.utils.data import DataLoader
from src.model import Model
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
from train import train
from validation import validation

#test
def test(model,criterion):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
        test_loss = test_loss/len(test_loader)
    return test_loss


def adjust_lr(epoch,optimizer):
    # optimize lr
    step = [10, 20, 30, 40]
    base_lr = 0.1
>>>>>>> efc809b30616d9a2f2ba5a4b88ce989dd4eee2b0
    lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr
    return lr


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
    #hyperparameters
    epoch = 100
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, nesterov=True, momentum=0.9)
    early_stopping = EarlyStopping(patience=5)
    running_loss_array = np.array([])
    best_loss = 100.0

    for epoch in range(1,epoch+1):
        adjust_lr(epoch,optimizer)

        running_loss= train(model,criterion,optimizer,train_loader)
        val_loss, val_loss_aray= validation(model,criterion,validation_loader)

>>>>>>> efc809b30616d9a2f2ba5a4b88ce989dd4eee2b0
        running_loss_array= np.append(running_loss_array, running_loss)
        early_stopping(val_loss, model)
        if val_loss < best_loss:
            best_loss = val_loss
<<<<<<< HEAD
            torch.save(model.state_dict(), f'../checkpoint/{timestamp}/checkpoint.pt')
=======
            torch.save(model.state_dict(), f'../checkpoint/checkpoint_{num_model}.pt')
>>>>>>> efc809b30616d9a2f2ba5a4b88ce989dd4eee2b0
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'epoch={epoch} , running loss={running_loss : .4f}')
<<<<<<< HEAD
    return running_loss_array,val_loss_aray,model

def multi_train():
    # train n times and get n models
    for ep in range(args.n):
        time_now = time.time()
        local_time = time.localtime(time_now)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        file_name = f"../checkpoint/{timestamp}"
        os.makedirs(file_name)
        print(f"The {local_time.tm_min} minute model's trainning process ")
        running_loss_array,val_loss_aray,model= train_val(timestamp,adjust_lr,inputs_size)
        # draw and save the images in each file
        loss_view(running_loss_array, val_loss_aray,timestamp)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)
        prediction_label_view(model, test_loader,timestamp)

    # model.load_state_dict(torch.load(f'../checkpoint/checkpoint_{best_model}.pt'))




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

def multi_train(inputs_size):
    # train ten times and get ten models, choose the best one
    criterion = torch.nn.MSELoss(reduction='mean')
    best_loss = 1
    best_model = 0
    model = Model(inputs_size)
    for num_model in range(1, 11):
        model = Model(inputs_size)
        print(f"The {num_model} model's trainning process ")
        main(num_model, model)
        test_loss = test(model, criterion)
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = num_model
    print(f"The best model is {best_model} with test loss {best_loss}")
    model.load_state_dict(torch.load(f'../checkpoint/checkpoint_{best_model}.pt'))

    prediction_label_view(model, test_loader)
    # loss_view(running_loss_array, val_loss_aray)

#load data
train_loader = DataLoader(dataset=dataset_train,batch_size=170,shuffle=True)
test_loader = DataLoader(dataset=dataset_test,batch_size=170,shuffle=False)
validation_loader = DataLoader(dataset=dataset_val,batch_size=170,shuffle=False)

if __name__ == '__main__':
    multi_train(inputs_size)
>>>>>>> efc809b30616d9a2f2ba5a4b88ce989dd4eee2b0
