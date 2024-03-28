
#training model
def train(model,criterion,optimizer,train_loader):
    model.train()
    running_loss = 0.0
    for _,data in enumerate(train_loader):
        inputs,labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    running_loss = running_loss/len(train_loader)
    return running_loss