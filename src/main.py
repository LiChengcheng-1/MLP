import torch
import numpy as np
from src.earlystopping import EarlyStopping
from src.data_process import dataset_train,dataset_test,dataset_val,inputs_size
from torch.utils.data import DataLoader
from src.model import Model
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
    lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr
    return lr


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

        running_loss_array= np.append(running_loss_array, running_loss)
        early_stopping(val_loss, model)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'../checkpoint/checkpoint_{num_model}.pt')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'epoch={epoch} , running loss={running_loss : .4f}')
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
