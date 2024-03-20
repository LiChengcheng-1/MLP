#数据准备

import torch
from torch.utils.data import Dataset,DataLoader
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class ExcelDataset(Dataset):
    def __init__(self,inputs,label):

        self.x_data = torch.from_numpy(inputs).float()
        self.y_data = torch.from_numpy(label).float()

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return len(self.x_data)

df = pandas.read_excel('../data/12_30 bis 13_30 Uhr Umf + Vega.xlsx', header=8, index_col=0, sheet_name=0)
df = df.replace("#-1", 0)
df = df[df["app1_q [l/s]"] != 0]
df = df.dropna()

#correlation analysis
correlation = df.corr(numeric_only=True)
correlation_label = df.corr(numeric_only=True).drop(["app1_q [l/s]"])["app1_q [l/s]"]
correlation_label_new = correlation_label.loc["p1_v1 [m/s]":"p1_v16 [m/s]"]
#choose Positive correlation data
need_remove = []
for index, value in correlation_label_new.items():
    if value <= 0:
        need_remove.append(index)
correlation_label_new = correlation_label_new.drop(need_remove)

# #Extract feature and labels
inputs = df[correlation_label_new.index].values
label = df.loc[:, "app1_q [l/s]"].values
label = label.reshape(-1,1)

#know the number of features
inputs_rows, inputs_size = inputs.shape

# data normalization
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

normalized_inputs =scaler_x.fit_transform(inputs)
normalized_label = scaler_y.fit_transform(label)
# cc=scaler_y.inverse_transform(c)

#devide data into train set validation set and test set
X_trainal, X_test, y_trainal, y_test = train_test_split(normalized_inputs, normalized_label, test_size=0.2, random_state=1)
X_train,X_validation,y_train,y_validation = train_test_split(X_trainal, y_trainal, test_size=0.25, random_state=1)
dataset_train = ExcelDataset(X_train,y_train)
dataset_test = ExcelDataset(X_test,y_test)
dataset_validation= ExcelDataset(X_validation,y_validation)
#instantiate dataset objects
train_loader = DataLoader(dataset=dataset_train,batch_size=170,shuffle=True,num_workers=2)
test_loader = DataLoader(dataset=dataset_test,batch_size=170,shuffle=True,num_workers=2)
validation_loader = DataLoader(dataset=dataset_validation,batch_size=170,shuffle=False,num_workers=2)

