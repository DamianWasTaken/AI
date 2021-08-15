#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Installing/Updating libraries
pip install numpy
pip install pandas
pip install pyplot
pip install seaborn
pip install sklearn
pip install keras
pip install tensorflow


# Bitcoin forecast using LSTM

# In[33]:


#Data manipulation libraries
import numpy as np
import pandas as pd
#Data visualisation libraries
from matplotlib import pyplot as plt
import seaborn as sns


# Now, pulling Bitcoin data from Yahoo finance(Historical Data)
# 

# In[34]:


#only 370 instances
bitcoinPath2= "https://raw.githubusercontent.com/DamianWasTaken/Python/main/BTC-USD.csv" 
#Used an user's Github BTC-USD upload instead of downloading and uploading it again from 1/10/2013 to 17/04/2021
bitcoinPath = "https://raw.githubusercontent.com/Milisha-gupta/Status-Neo/main/BTC_USD_2013-10-01_2021-04-17-CoinDesk.csv" 

btcData = pd.read_csv(bitcoinPath)

btcDataSorted = btcData.sort_values('Date')

#confirming date 
btcDataSorted.head()


# In[ ]:


#confirming end date 
btcDataSorted.tail()


# In[35]:


#Closing price will be the target of the prediction, as such it needs to be separated from the rest of the data
btcPrice = btcDataSorted[['Closing Price (USD)']]
#Setting the plot's parameters and plotting price
plt.plot(figsize = (15,9))
plt.plot(btcPrice)
plt.xticks(range(0, btcDataSorted.shape[0], 50), btcDataSorted['Date'].loc[::50], rotation=45)
plt.title('Bitcoin Price', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price(USD)', fontsize=18)
plt.show()


# In[6]:


#Looking at the instances and data type
btcPrice.info()


# Preparing Data

# In[36]:


#Normalising data, essentially putting all variables on the same scale 
#so that certain disproportionate variable's values don't influence the weights too much
from sklearn.preprocessing import MinMaxScaler
minMax = MinMaxScaler() #default value for scaling is 0-10, scaling should be adjusted depending on the activation function used
normalisedData =  minMax.fit_transform(btcPrice.values)

normalisedData


# Splitting data into training and testing, this needs to be done so as to evaluate the results obtained from 
# testing the network that was trained on the training data set by feeding it data that it has never seen before, 
# a much more effective and reliable way of splitting the data for this is bootstraping, 
# 
# the % depends entirely on how much data you have, if it's hundreds of thousands to millions of instances, apply a ratio of about 2-5% Testing and 95-98% Training, if less around 20-80% should do the trick, obviously this is dependent on research question and data at hand

# In[49]:


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
    
        data.append(np.reshape(dataset[indices],(history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)
    

#These values represent days, essentailly we're using 5 days 
#of time searies events to learn paterns in the data and predict the furute_target
past_history = 5
future_target = 0

#80-20% ratio split
trainSplit = int(len(normalisedData)* 0.8)


#Using univariate_data function to split the data into Train and Test
x_Train, y_Train = univariate_data(normalisedData,
                                0,
                                trainSplit,
                                past_history,
                                future_target)

x_Test, y_Test = univariate_data(normalisedData,
                                trainSplit,
                                None,
                                past_history,
                                future_target)


# LSTM Model Building
# 
# This is where we can do adjustments that will have the major impacts in efficiency, by adjusting the Hyper parameters, whilst there are a few theories into hyper parameters selection, it's highly dependent on the Data and use case, as such deciding empirically usually brings good results

# In[50]:


import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, LSTM, LeakyReLU, Dropout

#set the number or neurons, learning rate, function, optimiser function, loss function, batch and epochs, essentially hyper parameters
num_units = 64
learning_Rate = 0.0001
activation_function = 'sigmoid'
adam = Adam(learning_rate=learning_Rate)
loss_function = 'mse'
batch_size = 5
num_epochs = 250

#Calling on sequential and putting everything together
model = Sequential()
model.add(LSTM(units = num_units, activation = activation_function, input_shape = (None, 1)))
model.add(LeakyReLU(alpha = 0.5))
model.add(Dropout(0.1))
model.add(Dense(units = 1))



model.compile(optimizer = adam, loss=loss_function)


# In[16]:


model.summary()


# Training model

# In[51]:


history = model.fit(
    x_Train,
    y_Train,
    validation_split = 0.1,
    batch_size = batch_size,
    epochs = num_epochs,
    shuffle=False
)


# In[53]:


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure(figsize = (15,9))

plt.plot(epochs, loss, 'b', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title("Training and Validation Loss")
plt.legend()

plt.show()


# In[58]:


original = pd.DataFrame(minMax.inverse_transform(y_Test))
predictions = pd.DataFrame(minMax.inverse_transform(model.predict(x_Test)))
sns.set(rc={'figure.figsize':(11.7+2,8.27+2)})
ax = sns.lineplot(x=original.index, y=original[0], label='Test Data', color='royalblue')
ax = sns.lineplot(x=predictions.index, y=predictions[0], label='predictions', color='tomato')
ax.set_title('Bitcoins Price', size = 14, fontweight='bold')
ax.set_xlabel('Days', size=14)
ax.set_ylabel('Cost (USD)', size = 14)
ax.set_xticklabels('', size=10)

