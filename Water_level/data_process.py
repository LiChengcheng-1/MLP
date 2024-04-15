import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.dataset import MLPDataset

dataset = {}
for i in range(3):
    df = pandas.read_csv(f'data\Fluid{i}.csv')

    #Extract features and labels
    features = df.iloc[:,0:3].values
    labels = df.iloc[:, 3].values
    labels= labels.reshape(-1,1)

    #know the number of features
    inputs_rows, inputs_size = features.shape

    #devide data into train set validation set and test set
    X_train, X_rest, y_train, y_rest = train_test_split(features, labels, test_size=0.25)
    X_test,X_val,y_test,y_val = train_test_split(X_rest, y_rest, test_size=0.5)

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

    key = f"dataset_{i}"
    dataset[key] =[dataset_test,dataset_val,dataset_test]
