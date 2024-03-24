from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import zscore
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import os
import numpy as np
from sklearn import metrics
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

path = "./car_data/"
filename_read = os.path.join(path,"Sport car price.csv")
df = pd.read_csv(filename_read,na_values=['NA','?'])

def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

cars = df['Car Make']
cars = df['Car Model']
df.drop('Car Make',axis=1,inplace=True)
df.drop('Car Model',axis=1,inplace=True)

missing_median(df, 'Year')

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

x = df_normalized[['Horsepower', 'Torque (lb-ft)']].values  
y = df_normalized['Engine Size (L)'].values

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=45)

model = Sequential()
model.add(Dense(100, input_dim=x.shape[1], activation='relu'))
model.add(Dense(80))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=2,epochs=200)

pred = model.predict(x_test)
 
score = metrics.mean_squared_error(pred,y_test)
print("Final score (MSE): {}".format(score))

score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))

def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

chart_regression(pred.flatten(),y_test)

