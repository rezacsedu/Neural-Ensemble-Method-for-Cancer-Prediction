# coding: utf-8

import keras

import numpy as np
import gzip
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical

import os
import pandas as pd
import numpy as np
import sys
from time import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import TimeDistributed
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('tf')
import itertools
from keras.regularizers import L1L2

import numpy as np
import pandas as pd
np.random.seed(10)

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adagrad
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn import manifold
import keras.layers.normalization as bn

from sklearn.utils import shuffle
import os        
import keras

from sklearn.preprocessing import normalize  
from sklearn.preprocessing import normalize  
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from keras import regularizers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, UpSampling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Dense
from keras.layers import Flatten, Dense, Reshape
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D, RepeatVector
from keras.models import Model

import pandas

def prepareData(classification_type): 
    df = pd.read_csv("all_gene_input_data.csv", sep='\t')
    
    df = df.drop(df.columns[[0]], axis=1)
    
    if classification_type =='type':
        df = df[df['ctype'] != 15]
        for val in range(1, np.unique(df['ctype']).size + 1):
            df.ix[df['ctype'] == val, 'ctype'] = val - 1  # labels needs to start from 0
    
    if classification_type =='identification':
        df.ix[df['ctype'] != 15, 'ctype'] = 0
        df.ix[df['ctype'] == 15, 'ctype'] = 1

    column_headers = df.columns.values.tolist()
    column_headers.remove('ctype')
    
    features = df[column_headers].values
    labels = df['ctype'].values 
    
    normalizedFeatures = normalize(features.reshape(features.shape[0],-1), norm='max', axis=0).reshape(features.shape)
  
    print(type(normalizedFeatures))
    print(type(labels))

    #features = binary[cols]
    #labels = binary['ctype']

    features = np.array(normalizedFeatures)
    labels = np.array(labels)

    print(features.shape)
    print(labels.shape)

    #print(images.shape)
    features, labels = shuffle(features, labels, random_state = 0)  # shuffle the data
    print("Shauffle completed!")
    
    return features, labels


# In[3]:


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# In[4]:


def prepare_test_train_valid(features, labels):
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.25, random_state=12345)
    
    print('X_train shape:', train_x.shape)
    print('Y_train shape:', train_y.shape)
    
    train_x = np.reshape(train_x,(train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x,(test_x.shape[0], 1, test_x.shape[1]))
    
    return train_x, test_x, train_y, test_y


# In[5]:


type_features, type_labels = prepareData('type')


# In[125]:


labels = one_hot_encode(type_labels)
    
from sklearn.utils import class_weight
train_x, test_x, train_y, test_y = prepare_test_train_valid(type_features, labels)

sample_weights = class_weight.compute_sample_weight('balanced', train_y)
print(train_x.shape)
train_x = train_x.transpose(0,2,1)
test_x = test_x.transpose(0,2,1)

print(train_x.shape)

train_y_1 = to_categorical(train_y)
test_y_1 = to_categorical(test_y)

print(test_y.shape)


# In[51]:


train_X = np.expand_dims(train_x, axis=0)


# In[123]:


# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_x.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=test_x.shape))


# # The Convolutional Autoencoder!

# In[114]:


batch_size = 64
epochs = 200
inChannel = 1
x, y = 1, 20308
input_layer = Input(shape=(20308, 1))
#input_img = Input(shape = (x, y, inChannel))
num_classes = 14


# Let's create separate encoder and decoder functions since you will be using encoder weights later on for classification purpose!

# # Classification model

# In[115]:


def encoder(input_layer):
    conv1 = Conv1D(32, 2, activation='relu', padding='same')(input_layer) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(32, 2, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=1, strides=2)(conv1) #14 x 14 x 32
    conv2 = Conv1D(64, 2, activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(64, 2, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=1, strides=2)(conv2) #7 x 7 x 64
    conv3 = Conv1D(128, 2, activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(128, 2, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(256, 2, activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(256, 2, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    return conv4


# In[116]:


def decoder(conv4):    
    conv5 = Conv1D(128, 2, activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(128, 2, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(64, 2, activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(64, 2, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 =   UpSampling1D(2)(conv6) #14 x 14 x 64
    conv7 = Conv1D(32, 2, activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(32, 2, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 =   UpSampling1D(2)(conv7) # 28 x 28 x 32
    decoded = Conv1D(1, 2, activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    
    return decoded


# In[117]:


autoencoder = Model(input_layer, decoder(encoder(input_layer)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

autoencoder.summary()


# In[ ]:


autoencoder_train = autoencoder.fit(train_x, train_x, batch_size=32,epochs=1,verbose=1,validation_split=0.1)


