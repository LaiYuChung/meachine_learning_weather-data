#匯入套件
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import time
np.random.seed(0)

#載入一年資料
def data_input(dataname):
    data = pd.read_csv(dataname,index_col=0)
    data = data.iloc[:,[0,1,3,4,5,6,7,8,2]]

    X = data.iloc[:, :8].values  
    y = data.iloc[:, 8].values 
    X = X.reshape((X.shape[0], X.shape[1], 1))
    tt1 = [[range(160),range(160,201)],
      [range(41,201),range(201,242)],
      [range(82,242),range(242,283)],
      [range(123,283),range(283,324)],
      [range(164,324),range(324,365)]]
    tt1 = np.array(tt1)
    test_1i = [160,201,242,283,324]
    return data,X,y,tt1,test_1i

#載入五年資料
def data_input5(dataname):
    data = pd.read_csv(dataname,index_col=1)
    data = data.iloc[:,[1,2,4,5,6,7,8,9,3]]
    
    X = data.iloc[:, :8].values  
    y = data.iloc[:, 8].values 
    X = X.reshape((X.shape[0], X.shape[1], 1))
    tt1 = [[range(800),range(800,1005)],
      [range(205,1005),range(1005,1210)],
      [range(410,1210),range(1210,1415)],
      [range(615,1415),range(1415,1620)],
      [range(820,1620),range(1620,1826)]]
    tt1 = np.array(tt1)
    test_1i = [800,1005,1210,1415,1620]
    return data,X,y,tt1,test_1i

#載入十年資料
def data_input10(dataname):
    data = pd.read_csv(dataname,index_col=1)
    data = data.iloc[:,[1,2,4,5,6,7,8,9,3]]
    
    X = data.iloc[:, :8].values  
    y = data.iloc[:, 8].values 
    X = X.reshape((X.shape[0], X.shape[1], 1))
    tt1 = [[range(1600),range(1600,2010)],
      [range(410,2010),range(2010,2420)],
      [range(820,2420),range(2420,2830)],
      [range(1230,2830),range(2830,3240)],
      [range(1640,3240),range(3240,3652)]]
    tt1 = np.array(tt1)
    test_1i = [1600,2010,2420,2830,3240]
    return data,X,y,tt1,test_1i

#建立model
def create_cnn_model(layer,unit_list):
    if layer==1:
        model = Sequential()
        model.add(Conv1D(unit_list[0], 3, activation='relu')) 
        model.add(MaxPooling1D(pool_size = 2)) 
        model.add(Flatten()) 
        model.add(Dense(units=7))
    elif layer==2:
        model = Sequential()
        model.add(Conv1D(unit_list[0], 3, activation='relu')) 
        model.add(Conv1D(unit_list[1], 3, activation='relu')) 
        model.add(MaxPooling1D(pool_size = 2))
        model.add(Flatten()) 
        model.add(Dense(units=7))
    elif layer==3:
        model = Sequential()
        model.add(Conv1D(unit_list[0], 3, activation='relu')) 
        model.add(Conv1D(unit_list[1], 3, activation='relu')) 
        model.add(Conv1D(unit_list[2], 3, activation='relu')) 
        model.add(MaxPooling1D(pool_size = 2))
        model.add(Flatten()) 
        model.add(Dense(units=7))
    return model

#訓練模型，並將結果寫入txt檔
def training_all(data,X,y,tt1,test_1i,layer,unit_list):
    f = open("cnn_result.txt",mode="a")
    
    all_time = []
    all_error = []
    
    for i in range(len(tt1)):

        X_train, X_test = X[tt1[i][0]], X[tt1[i][1]]
        y_train, y_test = y[tt1[i][0]], y[tt1[i][1]]

        model = create_cnn_model(layer,unit_list)

        model.compile(optimizer='adam', loss='mse')
        start = time.time()  
        model.fit(X_train,y_train , epochs=50, batch_size=32, verbose=1)
        end = time.time()
        exe_time = round(end-start,2)
        all_time.append(round(exe_time,2))

        X_pred = X_test[-1:, :, :]
        y_pred = model.predict(X_pred)
        pre = list(y_pred[0])
        answer = y[test_1i[i]+7:test_1i[i]+14]
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
        data,X,y,tt1,test_1i = data_input("history_data_1_year")
        training_all(data,X,y,tt1,test_1i,1,i)
    elif len(i)==2:
        data,X,y,tt1,test_1i = data_input("history_data_1_year")
        training_all(data,X,y,tt1,test_1i,2,i)
    elif len(i)==3:
        data,X,y,tt1,test_1i = data_input("history_data_1_year")
        training_all(data,X,y,tt1,test_1i,3,i)

#跑五年資料
for i in num_of_neuron:
    if len(i)==1:
        data,X,y,tt1,test_1i = data_input("history_data_5_year")
        training_all(data,X,y,tt1,test_1i,1,i)
    elif len(i)==2:
        data,X,y,tt1,test_1i = data_input("history_data_5_year")
        training_all(data,X,y,tt1,test_1i,2,i)
    elif len(i)==3:
        data,X,y,tt1,test_1i = data_input("history_data_5_year")
        training_all(data,X,y,tt1,test_1i,3,i)

#跑十年資料
for i in num_of_neuron:
    if len(i)==1:
        data,X,y,tt1,test_1i = data_input("history_data_10_year")
        training_all(data,X,y,tt1,test_1i,1,i)
    elif len(i)==2:
        data,X,y,tt1,test_1i = data_input("history_data_10_year")
        training_all(data,X,y,tt1,test_1i,2,i)
    elif len(i)==3:
        data,X,y,tt1,test_1i = data_input("history_data_10_year")
        training_all(data,X,y,tt1,test_1i,3,i)
