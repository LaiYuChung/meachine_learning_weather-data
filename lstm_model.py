#匯入套件
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(0)

#載入一年資料
def year1dp(dataname):
    data = pd.read_csv(dataname,index_col=0)
    data = data.iloc[:,[0,1,3,4,5,6,7,8,2]]

    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    X = np.array(features)
    y = np.array(target)
    tt1 = [[range(160),range(160,201)],
      [range(41,201),range(201,242)],
      [range(82,242),range(242,283)],
      [range(123,283),range(283,324)],
      [range(164,324),range(324,365)]]
    tt1 = np.array(tt1)
    test_1i = [160,201,242,283,324]
    return X,y,tt1,test_1i,features,target

#載入五年資料
def year5dp(dataname):
    data = pd.read_csv(dataname,index_col=1)
    data = data.iloc[:,[1,2,4,5,6,7,8,9,3]]

    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    X = np.array(features)
    y = np.array(target)
    tt1 = [[range(800),range(800,1005)],
      [range(205,1005),range(1005,1210)],
      [range(410,1210),range(1210,1415)],
      [range(615,1415),range(1415,1620)],
      [range(820,1620),range(1620,1826)]]
    tt1 = np.array(tt1)
    test_1i = [800,1005,1210,1415,1620]
    return X,y,tt1,test_1i,features,target

#載入十年資料
def year10dp(dataname):
    data = pd.read_csv(dataname,index_col=1)
    data = data.iloc[:,[1,2,4,5,6,7,8,9,3]]

    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    X = np.array(features)
    y = np.array(target)
    tt1 = [[range(1600),range(1600,2010)],
      [range(410,2010),range(2010,2420)],
      [range(820,2420),range(2420,2830)],
      [range(1230,2830),range(2830,3240)],
      [range(1640,3240),range(3240,3652)]]
    tt1 = np.array(tt1)
    test_1i = [1600,2010,2420,2830,3240]
    return X,y,tt1,test_1i,features,target

#建立model
def create_lstm_model(input_shape,layer,unit_list):
    if layer ==1:
        model = Sequential()
        model.add(LSTM(units=unit_list[0], input_shape=input_shape))
        model.add(Dense(units=7)) 
        model.compile(optimizer='adam', loss='mean_squared_error')
        
    elif layer==2:
        model = Sequential()
        model.add(LSTM(units=unit_list[0], input_shape=input_shape,return_sequences=True))
        model.add(LSTM(units=unit_list[0], input_shape=input_shape))
        model.add(Dense(units=7)) 
        model.compile(optimizer='adam', loss='mean_squared_error')
        
    elif layer==3:
        model = Sequential()
        model.add(LSTM(units=unit_list[0], input_shape=input_shape,return_sequences=True))
        model.add(LSTM(units=unit_list[1], input_shape=input_shape,return_sequences=True))
        model.add(LSTM(units=unit_list[2], input_shape=input_shape))
        model.add(Dense(units=7)) 
        model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#訓練模型，並將結果寫入txt檔
def year1train(X,y,tt1,test_1i,features,target,layer,unit_list):
    f = open("lstm_result.txt",mode="a")
    all_time = []
    all_error = []
    for i in range(len(tt1)):

        X_train, X_test = X[tt1[i][0]], X[tt1[i][1]]
        y_train, y_test = y[tt1[i][0]], y[tt1[i][1]]

        timestep = 7

        X2, y2 = [], []
        for k in range(len(X_train) - timestep - 7):
            X2.append(X_train[k:k+timestep])
            y2.append(y_train[k+timestep:k+timestep+7])

        X2 = np.array(X2)
        y2 = np.array(y2)

        input_shape = (X2.shape[1], X2.shape[2])
        model = create_lstm_model(input_shape,layer,unit_list)
        start = time.time() 
        model.fit(X2,y2, epochs=50, batch_size=32, verbose=1)
        end = time.time()
        exe_time = round(end-start,2)
        all_time.append(exe_time)

        last_7_days = X_test[:7].reshape(1, timestep, -1)
        predicted_future = model.predict(last_7_days)
        pre = list(predicted_future[0])
        answer = target[test_1i[i]+7:test_1i[i]+14]
        v =0
        for ii,ai in zip(pre,answer):
            v+=abs(round((ii-ai)/ai,2))
        all_error.append(round(v/7,4))
    print(all_time,all_error)
    ind = all_error.index(min(all_error))
    print("[{},{}]".format(all_time[ind],round(all_error[ind]*100,2)),file=f)
    f.close()

num_of_neuron = [[32],[64],[128],[32,32],[64,64],[128,128],[32,32,32],[64,64,64],[128,128,128]]

#跑一年資料
for i in num_of_neuron:
    if len(i)==1:
        X,y,tt1,test_1i,features,target = year1dp('history_data_1_year')
        year1train(X,y,tt1,test_1i,features,target,1,i)
    elif len(i)==2:
        X,y,tt1,test_1i,features,target = year1dp('history_data_1_year')
        year1train(X,y,tt1,test_1i,features,target,2,i)
    elif len(i)==3:
        X,y,tt1,test_1i,features,target = year1dp('history_data_1_year')
        year1train(X,y,tt1,test_1i,features,target,3,i)

#跑五年資料
for i in num_of_neuron:
    if len(i)==1:
        X,y,tt1,test_1i,features,target = year1dp('history_data_5_year')
        year1train(X,y,tt1,test_1i,features,target,1,i)
    elif len(i)==2:
        X,y,tt1,test_1i,features,target = year1dp('history_data_5_year')
        year1train(X,y,tt1,test_1i,features,target,2,i)
    elif len(i)==3:
        X,y,tt1,test_1i,features,target = year1dp('history_data_5_year')
        year1train(X,y,tt1,test_1i,features,target,3,i)

#跑十年資料
for i in num_of_neuron:
    if len(i)==1:
        X,y,tt1,test_1i,features,target = year1dp('history_data_10_year')
        year1train(X,y,tt1,test_1i,features,target,1,i)
    elif len(i)==2:
        X,y,tt1,test_1i,features,target = year1dp('history_data_10_year')
        year1train(X,y,tt1,test_1i,features,target,2,i)
    elif len(i)==3:
        X,y,tt1,test_1i,features,target = year1dp('history_data_10_year')
        year1train(X,y,tt1,test_1i,features,target,3,i)