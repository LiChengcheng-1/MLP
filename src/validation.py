#validation
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
            running_loss_aray = np.append(running_loss_aray, batch_loss.item())
        running_loss = running_loss/len(validation_loader)
    return running_loss, running_loss_aray
