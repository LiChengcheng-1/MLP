import torch
import numpy as np
import matplotlib.pyplot as plt
from codes.earlystopping import EarlyStopping
from codes.dataset import train_loader,test_loader,validation_loader,inputs_size
from codes.model import Model


#training model
def train_model(model,criterion,optimizer):
    model.train()
    loss_running = 0.0
    for i,data in enumerate(train_loader,start=0):
        inputs,labels = data

        y_pred = model(inputs)
        loss = criterion(y_pred,labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        loss_running += loss.item()
    average_loss = loss_running/i
    return average_loss
#validation
def validation(model,criterion):
    model.eval()
    loss_validation = np.array([])
    validation_x, validation_y = next(iter(validation_loader))
    voutput = model(validation_x)
    v_loss = criterion(voutput , validation_y)
    loss_validation = np.append(loss_validation, v_loss.item())
    aver_v_loss = np.mean(loss_validation)
    return aver_v_loss, loss_validation

def test(model,criterion):
    model.eval()
    test_loss = 0
    prediction=np.array([])
    label =np.array([])
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            prediction=np.append(prediction,output)
            label=np.append(label,target)

        aver_test_loss = test_loss/len(test_loader)
        print(f'测试的平均损失为{aver_test_loss}')
    return prediction,label

def Visualize1(loss_running,loss_validation):
    plt.plot(loss_running, label='Loss')
    plt.plot(loss_validation, label='Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.show()

def Visualize2(prediction,label):
    plt.plot(prediction, label='prediction')
    plt.plot(label, label='label')
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.grid()
    plt.legend()
    plt.show()



def main():
    model = Model(inputs_size)
    criterion = torch.nn.MSELoss(reduction='mean')
    #hyperparameters
    best_loss = 1000.0
    epoch = 100
    # optimize lr
    step = [10, 20, 30, 40]
    base_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, nesterov=True, momentum=0.9)
    def adjust_lr(epoch):
        lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
        for params_group in optimizer.param_groups:
            params_group['lr'] = lr
        return lr

    early_stopping = EarlyStopping(patience=5)
    model.load_state_dict(torch.load('checkpoint.pt'))
    loss_running = np.array([])
    for epoch in range(1,epoch+1):
        adjust_lr(epoch)
        average_loss= train_model(model,criterion,optimizer)
        aver_v_loss, loss_validation= validation(model,criterion)
        loss_running= np.append(loss_running, average_loss)
        early_stopping(aver_v_loss, model)
        if aver_v_loss < best_loss:
            best_loss = aver_v_loss
            torch.save(model.state_dict(), 'checkpoint.pt')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'epoch={epoch} , running loss={average_loss : .4f}')
    prediction, label= test(model, criterion)
    return loss_running,loss_validation,prediction,label
if __name__ == '__main__':
    loss_running,loss_validation,prediction,label= main()
    Visualize1(loss_running,loss_validation)
    Visualize2(prediction,label)
