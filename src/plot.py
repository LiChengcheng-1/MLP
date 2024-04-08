import matplotlib.pyplot as plt
import numpy as np
import torch
from data_process import scaler_label

def loss_view(running_loss,val_loss,timestamp):
    plt.plot(running_loss, label='Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.savefig(f'../checkpoint/{timestamp}/loss_view')
    plt.close()

def prediction_label_view(model,test_loader,timestamp):
    prediction=np.array([])
    label =np.array([])
    with torch.no_grad():
        inputs,target= next(iter(test_loader))
        output = model(inputs)
        output =scaler_label.inverse_transform(output)
        target = scaler_label.inverse_transform(target)
        prediction=np.append(prediction,output)
        label=np.append(label,target)
    plt.plot(prediction[::5], label='prediction')
    plt.plot(label[::5], label='label')
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.grid()
    plt.legend()
    plt.savefig(f'../checkpoint/{timestamp}/prediction_label_view')
    plt.close()



