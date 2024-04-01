#validation
<<<<<<< HEAD
import numpy as np
import torch
def validation(model,criterion,validation_loader):
    model.eval()
    running_loss_aray = np.array([])
    running_loss=0.0
    with torch.no_grad():
        for _,data in enumerate(validation_loader):
            validation_x,validation_y = data
            val_y_pred = model(validation_x)
            batch_loss = criterion(val_y_pred , validation_y)
            running_loss += batch_loss.item()
            running_loss_aray = np.append(running_loss_aray, running_loss.item())
        running_loss = running_loss/len(validation_loader)
    return running_loss, running_loss_aray
=======
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
>>>>>>> efc809b30616d9a2f2ba5a4b88ce989dd4eee2b0
