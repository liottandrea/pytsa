# %% ENV
# import numpy as np
import pandas as pd
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

print("---Enviroment---")
# %load_ext version_information
%reload_ext version_information
%version_information os, pandas, matplotlib, numpy, datetime


# define a conversion function for the native timestamps in the csv file
def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))

# %% DESCRIPTION
print("---Description---")
print("Bitcoin price analysis")
print("Reference:")
print("https://www.kaggle.com/mczielinski/bitcoin-historical-data/data")

# %% INPUT
print("---Input---")
print('Data listing...')
print(os.listdir('../input/bitcoin-historical-data'))

# read in the data and apply our conversion function, this spits out
# a DataFrame with the DateTimeIndex already in place
print('Using bitstampUSD_1-min_data...')
data = pd.read_csv(
    '../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv',
    parse_dates=True, date_parser=dateparse, index_col=[0])

print('Total null open prices: %s' % data['Open'].isnull().sum())

print('top and bottom 5 rows')
print(data.head(5))
print(data.tail(5))

print('Turtle Strategy')
# The first thing we need are our trading signals. The Turtle strategy
# was based on daily data and they used to enter breakouts (new higher
# highs or new lower lows) in the 22-60 day range roughly.
# We are dealing with minute bars here so a 22 minute new high isn't
# much to get excited about. Lets pick an equivalent to 60 days then.
# They also only considered Close price so lets do the same...

signal_lookback = 60 * 24 * 60 # days * hours * minutes

# here's our signal columns
data['Buy'] = np.zeros(len(data))
data['Sell'] = np.zeros(len(data))

# this is our 'working out', you could collapse these into the .loc call
#  later on and save memory but I've left them in for debug purposes,
# makes it easier to see what is going on
data['RollingMax'] = data['Close'].shift(1).rolling(
    signal_lookback,
    min_periods=signal_lookback).max()

data['RollingMin'] = data['Close'].shift(1).rolling(
    signal_lookback,
    min_periods=signal_lookback).min()

data.loc[data['RollingMax'] < data['Close'], 'Buy'] = 1
data.loc[data['RollingMin'] > data['Close'], 'Sell'] = -1

# plot
fig,ax1 = plt.subplots(1,1)
ax1.plot(data['Close'])
y = ax1.get_ylim()
ax1.set_ylim(y[0] - (y[1]-y[0])*0.4, y[1])

ax2 = ax1.twinx()
ax2.set_position(matplotlib.transforms.Bbox([[0.125,0.1],[0.9,0.32]]))
ax2.plot(data['Buy'], color='#77dd77')
ax2.plot(data['Sell'], color='#dd4444')



# Import the dataset and encode the date
df = pd.read_csv('../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2018-03-27.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()

print("plot the time series")
fig,ax1 = plt.subplots(1,1)
ax1.plot(Real_Price)

# split data
prediction_days = 30
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]

# Data preprocess
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)



# Making the predictions
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)


# Visualising the results
plt.figure(figsize=(5,5), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'Real BTC Price')
plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction', fontsize=8)
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()
