import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.earlystopping import EarlyStopping
from data_process import dataset_train,dataset_test,dataset_val,inputs_size,scaler_output
from torch.utils.data import DataLoader
from model import Model
from plot import prediction_label_view,loss_view
from train import train
from validation import validation
import time
import argparse
from sklearn.metrics import mean_absolute_percentage_error,r2_score
import wandb


# test
def test(model,criterion,test_loader,scaler_output=scaler_output):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for _,data in enumerate(test_loader):
            inputs ,target = data
            output = model(inputs)

            #anti-normalization
            output =torch.tensor(scaler_output.inverse_transform(output)).float()
            target = torch.tensor(scaler_output.inverse_transform(target)).float()

            # evaluate
            mape_score = mean_absolute_percentage_error(target,output,multioutput="uniform_average")
            R2_score = r2_score(target,output,multioutput="uniform_average")

            loss = criterion(output, target)
            running_loss += loss.item()
        running_loss = running_loss/len(test_loader)
    return running_loss,mape_score,R2_score




def train_val(file_name,inputs_size,train_loader,validation_loader,args):

   #prepare the model,criterion,optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    model = Model(inputs_size=inputs_size, hidden_size=wandb.config["hidden_size"],hidden_layers=args.hidden_layers)
    optimizer = torch.optim.SGD(model.parameters(), wandb.config["lr"], nesterov=True, momentum=0.9)


    #start training and validation, record the loss
    early_stopping = EarlyStopping(patience=5)
    training_loss_array = np.array([])
    val_loss_array = np.array([])
    best_loss = 100.0


    for epoch in range(1,wandb.config["epoch"]+1):

        # Dynamically adjust learning rate
        step = [10, 20, 30, 40]
        base_lr = wandb.config["lr"]
        lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
        for params_group in optimizer.param_groups:
            params_group['lr'] = lr

        training_loss= train(model,criterion,optimizer,train_loader)
        wandb.log({"epoch_actual":epoch,"training_loss":training_loss})

        val_loss= validation(model,criterion,validation_loader)
        val_loss =round(val_loss, 5)
        wandb.log({"validation loss":val_loss})
        training_loss_array= np.append(training_loss_array, training_loss)
        val_loss_array= np.append(val_loss_array, val_loss)

        early_stopping(val_loss, model)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{file_name}/checkpoint.pt")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'epoch={epoch} , training loss={training_loss : .4f}')
        if epoch % 2 == 0:
            print(f'validation loss:{val_loss}')

    return training_loss_array,val_loss_array,model

def main(train_loader,validation_loader,test_loader,name,args):
    timestamp_list = []
    # train n times and get n models
    for ep in range(args.experiment_number):

        criterion = torch.nn.MSELoss(reduction='mean')


        # create the files with timestamp
        time_now = time.time()
        local_time = time.localtime(time_now)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        file_name = f"../{name}/checkpoint/lr_{args.lr} hidden_size_{args.hidden_size} batch_size_{args.batch_size}/{timestamp}"
        os.makedirs(file_name)

        #initialize the wandb project,start training and validation
        wandb.init(project=args.project_name,name= f"{timestamp}",config=args,reinit=True,mode="online")

        #train model
        print(f"The {local_time.tm_hour}clock {local_time.tm_min}minute {local_time.tm_sec}second model's trainning process ")
        training_loss_array,val_loss_aray,model= train_val(file_name,inputs_size,train_loader,validation_loader,args)

        # collect the model
        timestamp_list.append(timestamp)

        test_loss, mape_score, R2_score = test(model, criterion, test_loader, scaler_output=scaler_output)
        wandb.log({"test_loss": test_loss, "mape_score": mape_score, "R2_score": R2_score})
        with open(
                f"../{name}/checkpoint/lr_{args.lr} hidden_size_{args.hidden_size} batch_size_{args.batch_size}/{timestamp}/evaluate.txt",
                "a") as file:
            file.write(
                f"the test loss of this model is {test_loss},MAPE metric 's result is {mape_score},R2 score is {R2_score}")

        # draw and save the images in each file
        loss_view(training_loss_array, val_loss_aray,file_name)
        prediction_label_view(model, test_loader,file_name)

    return timestamp_list


if __name__ == '__main__':
    #hyperparameters
    parser = argparse.ArgumentParser(description="about hyperparameter")
    parser.add_argument("--epoch",type=int,default=100,help="epoch")
    parser.add_argument("--lr", type=float,default=0.1, help="learning rate")
    parser.add_argument("--hidden_size", type=int,default=20, help="hidden size")
    parser.add_argument("--batch_size", type=int,default=85, help="batch size")
    parser.add_argument("--experiment_number", type=int,default=2, help="experiment_number")
    parser.add_argument("--project_name", type=str,default="MLP_Water_speed", help="project_name")
    parser.add_argument("--mode", type=bool, default=True, help="Do you want Train model (True) or Test model(False)")
    parser.add_argument("--hidden_layers", type=int, default=2, help="the number of hidden layers of the model")
    args = parser.parse_args()

    # prepare the data
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)

    # prepare for test
    criterion = torch.nn.MSELoss(reduction='mean')
    model = Model(inputs_size=inputs_size, hidden_size=args.hidden_size,hidden_layers=args.hidden_layers)

    isTrain = args.mode
    if isTrain:

        # train and test
        timestamp_list = main(train_loader,validation_loader,test_loader,name="Water_speed",args=args)

        wandb.finish()

    else:
        #test
        name =input("type the name of model you want to test")
        timestamp = input("type the timestamp of the model you want to test")
        model = model.load_state_dict(torch.load(f"../Water_speed/checkpoint/{name}/{timestamp}/checkpoint.pt"))
        test_loss,mape_score,R2_score= test(model,criterion,test_loader)
        print(f"the testing loss is {test_loss}"
              f"the MAPE metrics' result is {mape_score} "
              f"the R2 metrics' result is {R2_score} "
              )
