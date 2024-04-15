import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import test,main
import torch
from data_process import dataset_train,dataset_test,dataset_val,inputs_size,scaler_output,dataset
from torch.utils.data import DataLoader
from src.model import Model
import argparse
import wandb






if __name__ == '__main__':

    #hyperparameters
    parser = argparse.ArgumentParser(description="about hyperparameter")
    parser.add_argument("--epoch",type=int,default=100,help="epoch")
    parser.add_argument("--lr", type=float,default=0.1, help="learning rate")
    parser.add_argument("--hidden_size", type=int,default=20, help="hidden size")
    parser.add_argument("--batch_size", type=int,default=85, help="batch size")
    parser.add_argument("--experiment_number", type=int,default=2, help="experiment_number")
    parser.add_argument("--project_name", type=str,default="MLP_Water_level", help="project_name")
    parser.add_argument("--mode", type=bool, default=True, help="Do you want Train model (True) or Test model(False)")
    parser.add_argument("--hidden_layers", type=int, default=2, help="the number of hidden layers of the model")
    args = parser.parse_args()

    # # prepare the data
    # for i in range(3):
    #     dataset_test= dataset[f"dataset_{i}"][dataset_test]
    #     dataset_val= dataset[f"dataset_{i}"][dataset_val]
    #     dataset_train= dataset[f"dataset_{i}"][dataset_train]

    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)

    #prepare for test
    criterion = torch.nn.MSELoss(reduction='mean')
    model = Model(inputs_size=inputs_size, hidden_size=args.hidden_size,hidden_layers=args.hidden_layers)

    isTrain = args.mode
    if isTrain:
        # train and test
        timestamp = main(train_loader,validation_loader,test_loader,name="Water_level",args=args)
        wandb.finish()

    else:
        #test
        name =input("type the name of model you want to test")
        timestamp = input("type the timestamp of the model you want to test")
        model = model.load_state_dict(torch.load(f"../Water_level/checkpoint/{name}/{timestamp}/checkpoint.pt"))
        test_loss,mape_score,R2_score= test(model,criterion,test_loader,scaler_output)
        print(f"the testing loss is {test_loss}"
              f"the MAPE metrics' result is {mape_score} "
              f"the R2 metrics' result is {R2_score} "
              )