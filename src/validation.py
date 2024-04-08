#validation
import numpy as np
import torch
def validation(model,criterion,validation_loader):
    model.eval()
<<<<<<< HEAD
    val_loss=0.0
=======
    running_loss_aray = np.array([])
    running_loss=0.0
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
    with torch.no_grad():
        for _,data in enumerate(validation_loader):
            validation_x,validation_y = data
            val_y_pred = model(validation_x)
<<<<<<< HEAD
            loss = criterion(val_y_pred , validation_y)
            val_loss += loss.item()
        val_loss = val_loss/len(validation_loader)
    return val_loss
=======
            batch_loss = criterion(val_y_pred , validation_y)
            running_loss += batch_loss.item()
            running_loss_aray = np.append(running_loss_aray, batch_loss.item())
        running_loss = running_loss/len(validation_loader)
    return running_loss, running_loss_aray
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
