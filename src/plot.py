import matplotlib.pyplot as plt
import numpy as np
import torch

def loss_view(running_loss,val_loss):
    plt.plot(running_loss, label='Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.show()

def prediction_label_view(model,test_loader):
    prediction=np.array([])
    label =np.array([])
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            prediction=np.append(prediction,output)
            label=np.append(label,target)
    plt.plot(prediction, label='prediction')
    plt.plot(label, label='label')
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.grid()
    plt.legend()
    plt.show()



