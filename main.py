import torch
from torch.utils.data import Dataset,DataLoader
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler #归一化


#数据准备
# 加载数据库
df = pandas.read_excel('E:/lesson/MLP/12_30 bis 13_30 Uhr Umf + Vega.xlsx', header=8, index_col=0, sheet_name=0)
df = df.replace("#-1", 0)

# inputs = df.loc[:,"p1_v1 [m/s]":"p1_v16 [m/s]"].values
# label = df.loc[:,"app1_q [l/s]"].values
#
# # #删除值为0的行
# # zero_line = [1]
# # for index, data in enumerate(label,start=1):
# #     if data == 0:
# #         zero_line.append(index)
# #
# # try:
# #     df_new = df.drop(zero_line)
# # except:
# #     pass

df = df[df["app1_q [l/s]"] != 0]
df = df.dropna()
inputs = df.loc[:, "p1_v1 [m/s]":"p1_v16 [m/s]"].values
label = df.loc[:, "app1_q [l/s]"].values

X_train, X_test, y_train, y_test = train_test_split(inputs, label, test_size=0.25, random_state=1)


class ExcelDataset(Dataset):
    def __init__(self,inputs,label):

        self.x_data = torch.from_numpy(inputs).float()
        self.y_data = torch.from_numpy(label).float()

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return len(self.x_data)

dataset_train = ExcelDataset(X_train,y_train)
dataset_test = ExcelDataset(X_test,y_test)

train_loader = DataLoader(dataset=dataset_train,batch_size=341,shuffle=True,num_workers=2)
test_loader = DataLoader(dataset=dataset_test,batch_size=341,shuffle=True,num_workers=2)


#定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(16, 100)
        self.linear2 = torch.nn.Linear(100, 100)
        self.linear3 = torch.nn.Linear(100, 100)
        self.linear4 = torch.nn.Linear(100, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.linear4(x)
        return x

model = Model()

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)





loss_validation = np.array([])
loss_running_view =np.array([])
#迭代更新
if __name__ == '__main__':
    for epoch in range(100):
        loss_running = 0.0
        for i,data in enumerate(train_loader,start=0):
            #data
            inputs,labels = data
            #前馈
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            #反馈
            optimizer.zero_grad()
            loss.backward()
            #更新
            optimizer.step()
            loss_running += loss.item()

        print(f'epoch={epoch} , running loss={loss_running : .4f}')
        validation_x, validation_y = next(iter(test_loader))
        voutput = model(validation_x)
        v_loss = criterion(voutput , validation_y)
        loss_validation = np.append(loss_validation, v_loss.item())
        loss_running_view = np.append(loss_running_view, loss_running/i)

#保存参数信息
path = './model.pth'
torch.save(model.state_dict(), path)
# #加载参数
# model.load_state_dict(torch.load(path))

#可视化
plt.plot(loss_running_view , label = 'Loss')
plt.plot(loss_validation, label = 'Validation Loss')
plt.grid()
plt.legend()
plt.show()