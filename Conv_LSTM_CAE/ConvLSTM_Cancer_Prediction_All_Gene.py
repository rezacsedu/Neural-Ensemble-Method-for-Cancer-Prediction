import os
import pandas as pd
import numpy as np
import sys
from time import time
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten
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
from sklearn.preprocessing import normalize  
from sklearn.preprocessing import normalize  
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Dense
from keras.layers import Flatten, Dense, Reshape
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D
from keras.models import Model

def prepareData(classification_type): 
    df = pd.read_csv("/home/rkarim/Conv_LSTM_CNV/all_gene_input_data.csv", sep='\t')
    
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

    features = np.array(normalizedFeatures)
    labels = np.array(labels)

    print(features.shape)
    print(labels.shape)

    #print(images.shape)
    features, labels = shuffle(features, labels, random_state = 0)  # shuffle the data
    print("Shauffle completed!")
    
    return features, labels

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def prepare_test_train_valid(features, labels):
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.25, random_state=12345)
   
    print('X_train shape:', train_x.shape)
    print('Y_train shape:', train_y.shape)
    
    train_x = np.reshape(train_x,(train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x,(test_x.shape[0], 1, test_x.shape[1]))
    
    return train_x, test_x, train_y, test_y

def Conv_LSTM(num_classes, timesteps, reg):
    input_layer = Input(shape=(1, 20308))
    
    conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)    
    conv2 = Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', padding='same')(conv1)
    
    lstm1 = LSTM(128, return_sequences=True)(conv2)
    
    flat = Flatten()(lstm1)
    do = Dropout(0.7)(flat)
    output_layer = Dense(num_classes, activation='softmax')(do)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def model_train(model, number_epoch, train_x, train_y, classification_type):   
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if classification_type == 'type':
        loss = 'categorical_crossentropy'
    
    else:
        loss = 'binary_crossentropy'

    model.compile(loss=loss, metrics=['accuracy'], optimizer=adam)
    tensorboardRNN = TensorBoard(log_dir="RNN_logs/{}".format(time()))
    
    #for i in range(number_epoch):
    history1 = model.fit(train_x, train_y, validation_split=0.1, callbacks=[tensorboardRNN], batch_size=32, epochs=int(number_epoch), shuffle=False)
    #model.reset_states()        
    
    print(model.summary())

    return model, history1

def model_evaluate(rnn_model, classification_type, test_x, test_y):            
    y_prob = rnn_model.predict(test_x) 
    #rnn_model.reset_states()  
    
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(test_y, 1)

    # evaluate the model
    score, accuracy = rnn_model.evaluate(test_x, test_y, batch_size=32)
    #rnn_model.reset_states()
    
    print("Accuracy = {:.2f}".format(accuracy))
    print("Score = {:.2f}".format(score))

    # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
    if classification_type == 'type':
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        class_names = ["COAD", "GBM", "KIRC", "LGG", "LUAD", "LUSC", "OV", "UCEC", "BRCA", "HNSC","THCA", "PRAD", "STAD", "BLCA"]
        
        print ("F-Score:", round(f,3))
        print ("Precision:", round(p,3))
        print ("Recall:", round(r,3))
        print ("F-Score:", round(f,3)) 

        #_test_data_class(y_true, y_pred, class_names)
        import gc; gc.collect()
    
        print(classification_report(y_true, y_pred, target_names=class_names))
    
type_features, type_labels = prepareData('type')

def cancerTypeClassifier_ConvLSTM(classification_type, number_epoch, features, labels, reg):

    labels = one_hot_encode(labels)
    train_x, test_x, train_y, test_y = prepare_test_train_valid(features, labels)
    print(train_x.shape)
    print(test_x.shape) 

    timesteps = 128
    num_classes = 14

    model = Conv_LSTM(num_classes, timesteps, reg)

    trainedModel, history1 = model_train(model, number_epoch, train_x, train_y, classification_type)
    #plotLoses(history1, number_epoch)
    model_evaluate(trainedModel, classification_type, test_x, test_y) 
    
reg = L1L2(l1=0.10, l2=0.10)
cancerTypeClassifier_ConvLSTM('type', 20, type_features, type_labels, reg)