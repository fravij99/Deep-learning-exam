import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import hyperopt 
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, GRU
from hyperopt import hp, tpe, Trials, fmin, rand
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import time
import threading
from sklearn.datasets import load_iris
from numba import jit
import seaborn as sns


#DATA LOADING
excel=pd.read_excel(r"C:\Users\Francesco\OneDrive\Desktop\esami\deep learning\esame\data_bresso.xlsx")

df=pd.DataFrame(data=excel)

df2=pd.DataFrame(data=excel)
df['dateTime']=df['dateTime']-df['dateTime'][0]

df.drop(df.head(7000).index, inplace=True)

df_small=df.head(2000)


#RESCALING
df['cloudbase']=df['cloudbase']/df['cloudbase'].max()
df['outHumidity']=df['outHumidity']/df['outHumidity'].max()
df2['cloudbase']=df2['cloudbase']/df2['cloudbase'].max()
df2['outHumidity']=df2['outHumidity']/df2['outHumidity'].max()
df['outTemp']=df['outTemp']/df['outTemp'].max()
df2['outTemp']=df2['outTemp']/df2['outTemp'].max()



#PCA
plt.plot(df_small['dateTime'], df_small['barometer'])
plt.plot(df_small['dateTime'], df_small['outTemp'])
plt.plot(df_small['dateTime'], df_small['rain'])
plt.plot(df_small['dateTime'], df_small['outHumidity'])
plt.plot(df_small['dateTime'], df_small['windSpeed'])
plt.grid(True)
plt.title("Comparison",fontsize=15)
plt.xlabel("Time",fontsize=15)
plt.ylabel("Intensity",fontsize=15)
plt.legend(['barometer', 'outtemp', 'rain, ', 'humidity', 'windspeed'])
plt.show()


plt.plot(df_small['dateTime'], df_small['cloudbase'])
plt.xlabel("Time",fontsize=15)
plt.ylabel("Intensity",fontsize=15)
plt.title('cloudbase ')
plt.grid(True)
plt.show()


plt.plot(df_small['dateTime'], df_small['windSpeed'])
plt.plot(df_small['dateTime'], df_small['rain'])
plt.xlabel("Time",fontsize=15)
plt.ylabel("Intensity",fontsize=15)
plt.title('rain speed ')
plt.grid(True)
plt.show()


plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(2, 2)

ax[0, 0].scatter(df_small['outHumidity'], df_small['cloudbase'], s=20, c='r')
ax[0, 0].set_xlabel('outHumidity')
ax[0, 0].set_ylabel('cloudbase')
ax[0, 0].grid(True)
    
ax[0, 1].scatter(df_small['outTemp'], df_small['cloudbase'], s=20, c='g')
ax[0, 1].set_xlabel('outTemp')
ax[0, 1].grid(True)
ax[0, 1].set_ylabel('cloudbase')

ax[1, 0].scatter(df_small['outHumidity'], df_small['outTemp'], s=20, c='b')
ax[1, 0].set_xlabel('outHumidity')
ax[1, 0].set_ylabel('outTemp')
ax[1, 0].grid(True)

ax[1, 1].scatter(df_small['windSpeed'], df_small['outHumidity'], s=20, c='orange')
ax[1, 1].set_xlabel('windSpeed')
ax[1, 1].set_ylabel('outHumidity')
ax[1, 1].grid(True)
plt.show()

plt.figure(figsize=(5,5))

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(df.corr(), vmin=-1, vmax=1, 
                 cmap="coolwarm", 
                 ax=ax)
plt.show()



#CREATING MODELS 
start=time.time()

xtrain=df['outHumidity'].head(16000)
ytrain=df['cloudbase'].head(16000)

xtest=df2['outHumidity'].head(1500)
ytest=df2['cloudbase'].head(1500)

lr=0.01
batch=32
epochs=10
opt = tf.keras.optimizers.Adam(learning_rate=lr)

def LSTM_model():
    regressor=tf.keras.models.Sequential()
    regressor.add(GRU(units = 50, return_sequences = True, input_shape = (1, 1)))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 5, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))
    return regressor


#FITTING CLOUDBASE IN FUNCTION OF HUMIDITY
shape=1
model=LSTM_model() 
model.compile(optimizer = opt, loss = 'mean_squared_error')

hist=model.fit(xtrain, ytrain, epochs = epochs, batch_size = batch)

ypredict=model.predict(xtest)

plot_history(hist)

fig, ax = plt.subplots(1, 2)

ax[0].scatter(df2['outHumidity'].head(1500), df2['cloudbase'].head(1500), color='b')
ax[0].plot(xtest, ypredict, color='r')
ax[0].set_xlabel("Humindity",fontsize=15)
ax[0].set_ylabel("cloudbase",fontsize=15)
ax[0].grid(True)
ax[0].legend(['train values', 'test prediction'])

    
ax[1].plot(df2['cloudbase'].head(1500), color='b')
ax[1].plot(ypredict, color='r')
ax[1].set_xlabel("index",fontsize=15)
ax[1].set_ylabel("cloudbase",fontsize=15)
ax[1].grid(True)
ax[1].legend(['train values', 'test prediction'])

plt.show()


#uSING OTHER VARIABLES TO PREDICT HUMIDITY

df3=df
df4=df2


ytrain=df['outHumidity'].head(16000)
ytest=df2['outHumidity'].head(1500)

df3.drop(columns=['dateTime', 'Date', 'Time', 'barometer', 'humidex', 'rain', 'windDir', 'outHumidity'], axis=1, inplace=True)
df4.drop(columns=['dateTime', 'Date', 'Time', 'barometer', 'humidex', 'rain', 'windDir', 'outHumidity'], axis=1, inplace=True)

def LSTM_model2():
    regressor=tf.keras.models.Sequential()
    regressor.add(GRU(units = 50, return_sequences = True, input_shape = (3, 1)))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))
    return regressor

xtrain=df3.head(16000)
xtest=df4.head(1500)
lr=0.01
shape=3
model=LSTM_model2()
model.compile(optimizer = opt, loss = 'mean_squared_error')
hist=model.fit(xtrain, ytrain, epochs = epochs, batch_size = batch)

ypredict=model.predict(xtest)

plot_history(hist)
plt.show()

print(ypredict)

plt.plot(ytest, color='b')
plt.plot(ypredict, color='r')
plt.xlabel("index",fontsize=15)
plt.ylabel("Humidity",fontsize=15)
plt.title('predictions')
plt.grid(True)
plt.legend(['train values', 'test prediction'])
plt.show()

#TRYING TO USE ALLL THE OTHER PARAMETERS TO PREDICT HUMIDITY

def LSTM_model_tot():
    regressor=tf.keras.models.Sequential()
    regressor.add(GRU(units = 50, return_sequences = True, input_shape = (7, 1)))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units = 50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))
    return regressor

xtrain=df.head(16000)
xtest=df2.head(1500)
lr=0.01
shape=9

print(xtrain)
model=LSTM_model_tot()
model.compile(optimizer = opt, loss = 'mean_squared_error')
hist=model.fit(xtrain, ytrain, epochs = epochs, batch_size = batch)

ypredict=model.predict(xtest)

plot_history(hist)
plt.show()

print(ypredict)

plt.plot(ytest, color='b')
plt.plot(ypredict, color='r')
plt.xlabel("index",fontsize=15)
plt.ylabel("Humidity",fontsize=15)
plt.title('predictions')
plt.grid(True)
plt.legend(['train values', 'test prediction'])
plt.show()

end=time.time()
print(end-start, 's')

#hiperparameters optimizations
