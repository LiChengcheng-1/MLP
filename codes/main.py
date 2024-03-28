import torch
import numpy as np
from codes.earlystopping import EarlyStopping
from data.data_process import dataset_train,dataset_test,dataset_val,inputs_size
from torch.utils.data import DataLoader
from codes.model import Model
from utility import loss_view,prediction_label_view,adjust_lr


#load data
train_loader = DataLoader(dataset=dataset_train,batch_size=170,shuffle=True)
test_loader = DataLoader(dataset=dataset_test,batch_size=170,shuffle=False)
validation_loader = DataLoader(dataset=dataset_val,batch_size=170,shuffle=False)



#training model
def train_model(model,criterion,optimizer,train_loader):
    model.train()
    running_loss = 0.0
    for _,data in enumerate(train_loader):
        inputs,labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    running_loss = running_loss/len(train_loader)
    return running_loss

#validation
def validation(model,criterion,validation_loader):
    model.eval()
    val_loss_aray = np.array([])
    val_loss=0.0
    for _,data in enumerate(validation_loader):
        validation_x,validation_y = data
        val_y_pred = model(validation_x)
        loss = criterion(val_y_pred , validation_y)
        val_loss += loss.item()
        val_loss_aray = np.append(val_loss_aray, loss.item())
    val_loss = val_loss/len(validation_loader)
    return val_loss, val_loss_aray

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

def main(num_model,model):
    criterion = torch.nn.MSELoss(reduction='mean')
    #hyperparameters
    epoch = 100
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, nesterov=True, momentum=0.9)
    early_stopping = EarlyStopping(patience=5)
    running_loss_array = np.array([])
    best_loss = 100.0

    for epoch in range(1,epoch+1):
        adjust_lr(epoch,optimizer)
        running_loss= train_model(model,criterion,optimizer)
        val_loss, val_loss_aray= validation(model,criterion)

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

if __name__ == '__main__':
    multi_train(inputs_size)
