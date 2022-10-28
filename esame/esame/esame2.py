import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import hyperopt 
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout
from hyperopt import hp, tpe, Trials, fmin, rand
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import time
import threading
from sklearn.datasets import load_iris
from numba import jit, cuda
import seaborn as sns



excel=pd.read_excel(r"C:\Users\Francesco\OneDrive\Desktop\esami\deep learning\esame\data_bresso.xlsx", index_col=None)

df=pd.DataFrame(data=excel)

df2=pd.DataFrame(data=excel)
df['dateTime']=df['dateTime']-df['dateTime'][0]

df.drop(df.head(7000).index, inplace=True)

df_small=df.head(2000)


df['cloudbase']=df['cloudbase']/df['cloudbase'].max()
df['outHumidity']=df['outHumidity']/df['outHumidity'].max()
df2['cloudbase']=df2['cloudbase']/df2['cloudbase'].max()
df2['outHumidity']=df2['outHumidity']/df2['outHumidity'].max()
df['outTemp']=df['outTemp']/df['outTemp'].max()
df2['outTemp']=df2['outTemp']/df2['outTemp'].max()
df['humidex']=df['humidex']/df['humidex'].max()
df2['humidex']=df2['humidex']/df2['humidex'].max()
df['barometer']=df['barometer']/df['barometer'].max()
df2['barometer']=df2['barometer']/df2['barometer'].max()
df['rain']=df['rain']/df['rain'].max()
df2['rain']=df2['rain']/df2['rain'].max()
df['windDir']=df['windDir']/360
df2['windDir']=df2['windDir']/360
df['windSpeed']=df['windSpeed']/df['windSpeed'].max()
df2['windSpeed']=df2['windSpeed']/df2['windSpeed'].max()


df.dropna(axis='columns')
df2.dropna(axis='columns')
df.drop(columns=['dateTime', 'Date', 'Time', 'windDir'], axis=1, inplace=True)
df2.drop(columns=['dateTime', 'Date', 'Time', 'windDir'], axis=1, inplace=True)

start=time.time()


def test(model, features, labels):
    acc = model.evaluate(features, labels)
    return acc

@jit
def train(features, labels, parameters):
    model=tf.keras.models.Sequential()
    model.add(LSTM(units=parameters['layer_size'], input_shape=(7, 1)))
    model.add(Dense(1))
    adam=tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate'])
    model.compile(optimizer=adam, loss='mean_squared_error')
    model.fit(features, labels, epochs=5)
    return model

@jit
def finding_best(search_space, trial):    
    return fmin(fn=hyper_func, space=search_space, algo=tpe.suggest, max_evals=10, trials=trial)

xtrain=np.array(df.head(6000))
xtest=np.array(df2.head(1500))
ytrain=np.array(df['outHumidity'].head(6000))
ytest=np.array(df2['outHumidity'].head(1500))

print(type(xtest))
lr=0.001
shape=7
test_setup={'layer_size':2, 'learning_rate':1.0}

model=train(xtrain, ytrain, test_setup)
print('Loss model on test set is: ', test(model, xtest, ytest))

@jit(parallel=True)
def hyper_func(params):

     model = train(xtrain, ytrain, params)
     test_acc = test(model, xtest, ytest)
     return{'loss': -test_acc, 'status': STATUS_OK}

search_space={
              'layer_size':hp.choice('layer_size', np.arange(10, 100, 20)), 
              'learning_rate': hp.loguniform('learning_rate', -10, 0)}

trial=Trials()

best=finding_best(search_space, trial)##

print(space_eval(search_space, best))

_, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
xs = [t['tid'] for t in trial.trials]
ys = [-t['result']['loss'] for t in trial.trials]
ax1.set_xlim(xs[0]-1, xs[-1]+1)
ax1.scatter(xs, ys, s=20)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Accuracy')
xs = [t['misc']['vals']['layer_size'] for t in trial.trials]
ys = [-t['result']['loss'] for t in trial.trials]

ax2.scatter(xs, ys, s=20)
ax2.set_xlabel('Layers')
ax2.set_ylabel('Accuracy')

xs = [t['misc']['vals']['learning_rate'] for t in trial.trials]
ys = [-t['result']['loss'] for t in trial.trials]
ax3.scatter(xs, ys, s=20)
ax3.set_xlabel('learning_rate')
ax3.set_ylabel('Accuracy')
plt.show()

end=time.time()
print(end-start, 's')