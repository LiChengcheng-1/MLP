import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from dataset import MLPDataset
import numpy as np

#according to the EDA's visualization, we process the data
df = pandas.read_excel('../Water_speed/data/12_30 bis 13_30 Uhr Umf + Vega.xlsx')
df.columns = df.iloc[7]

#delete the missing value
selected_df = df.dropna()

#replace the #-1 with 0
selected_df = selected_df[selected_df['app1_q [l/s]'] != 0].replace("#-1",0)
selected_df =pandas.concat([selected_df.loc[8:,"app1_q [l/s]"],selected_df.loc[8:,"p1_v1 [m/s]":"p1_v16 [m/s]"]],axis=1)

#choose the most relevant features
correlation = selected_df.corr().drop(['app1_q [l/s]'])['app1_q [l/s]']
need_remove = []
for index in correlation.index:
    if correlation[index]< 0:
        need_remove.append(index)
corr_selected_df = selected_df.drop(need_remove,axis=1)

# turn type from object to float
corr_selected_df_float =corr_selected_df.astype(np.float64)

#Extract features and labels
features = corr_selected_df_float.iloc[:,1:].values
labels = corr_selected_df_float.iloc[:, 0].values
labels= labels.reshape(-1,1)

#know the number of features
inputs_rows, inputs_size = features.shape

#devide data into train set validation set and test set
X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)
X_train,X_val,y_train,y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=0.25, random_state=1)

# data normalization
scaler_input = MinMaxScaler(feature_range=(0, 1))
scaler_output = MinMaxScaler(feature_range=(0, 1))

scaler_input.fit(X_train)
scaler_output.fit(y_train)

X_train =scaler_input.transform(X_train)
y_train =scaler_output.transform(y_train)
X_val =scaler_input.transform(X_val)
y_val =scaler_output.transform(y_val)
X_test =scaler_input.transform(X_test)
y_test =scaler_output.transform(y_test)


#instante dataset
dataset_train = MLPDataset(X_train,y_train)
dataset_val= MLPDataset(X_val,y_val)
dataset_test = MLPDataset(X_test,y_test)

