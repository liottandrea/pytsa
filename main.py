#%% ENV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# extra functions
import dataInOut as myio
import gfkInOut as gfk
import tsa as mytsa
import dfEssential as mydf
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


#%% DESCR
'''
IntProj with Python POC
just try LSTM from tensorflow, veeery simple exercise
https://machinelearningmastery.com/\
time-series-forecasting-long-short-term-memory-network-python/
'''

#%% SETTING
country_key = 826
domain_product_group_name = "SMARTPHONES"

validation_step = forecast_step = 24

#%% INPUT
# retrieve actual
df = gfk.dpr_retrieveActual(country_key, domain_product_group_name)

# check missing value
df_missing = mydf.df_checkMissing(df)

#%% PREP
df.index = pd.to_datetime(df["DateKey"].values,format = "%Y%m%d")

# the index of the df has to be the date
series = mytsa.df_col2Series(df,"TotalUnitsSold")

#%% MAIN
# split data
series_train, series_test = mydf.df_splitDataset(series,forecast_step) 


# Data preprocess
training_set = series_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size=5, epochs=100)

# Making the predictions
test_set = series_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_tus = regressor.predict(inputs)
predicted_tus = sc.inverse_transform(predicted_tus)


#%% OUTPUT
# Visualising the results
plt.figure(figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()
plt.plot(test_set, color='red', label='Real BTC Price')
plt.plot(predicted_tus, color='blue', label='Predicted BTC Price')
plt.title('Predicted tus', fontsize=8)
df_test = series_test.reset_index()
x = df_test.index
labels = df_test.index
plt.xticks(x, labels, rotation='vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(8)
plt.xlabel('Time', fontsize=8)
plt.ylabel('TUS', fontsize=8)
plt.legend(loc=2, prop={'size': 8})
plt.show()
