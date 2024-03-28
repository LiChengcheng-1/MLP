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
