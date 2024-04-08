#training model
def train(model,criterion,optimizer,train_loader):
    model.train()
    running_loss = 0.0
    for _,data in enumerate(train_loader):
        inputs,labels = data
        y_pred = model(inputs)
        batch_loss = criterion(y_pred,labels)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        running_loss += batch_loss.item()
    running_loss = running_loss/len(train_loader)
    return running_loss


