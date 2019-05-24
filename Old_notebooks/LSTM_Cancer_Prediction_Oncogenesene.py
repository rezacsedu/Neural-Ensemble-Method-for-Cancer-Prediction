
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import sys
from time import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('tf')
import matplotlib.pyplot as plt
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

from sklearn.metrics import confusion_matrix
from confusion_matrix_pretty_print import _test_data_class
from sklearn.utils import shuffle

from sklearn.preprocessing import normalize  
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from keras import regularizers

# Set random seed
np.random.seed(0)

# In[2]:


def prepareData(classification_type): 
    df = pd.read_csv("oncogene_input_data1.csv", sep='\t')
    
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


# In[15]:


def prepare_test_train_valid(features, labels):
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.20, random_state=12345)
    test_x, valid_x, test_y, valid_y = train_test_split(train_x, train_y, test_size=0.10, random_state=12345)
    
    print('X_train shape:', train_x.shape)
    print('Y_train shape:', train_y.shape)
    
    train_x = np.reshape(train_x,(train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x,(test_x.shape[0], 1, test_x.shape[1]))
    valid_x = np.reshape(valid_x,(valid_x.shape[0], 1, valid_x.shape[1]))
    
    return train_x, test_x, train_y, test_y, valid_x, valid_y


# In[5]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[6]:


def model_build(num_classes, timesteps, data_dim, reg): #OK
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(timesteps, data_dim), recurrent_regularizer= reg)) 
    model.add(Dropout(0.50))
    
    # apply softmax to output
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


# In[16]:


def model_train(model, number_epoch, train_x, train_y, valid_x, valid_y, classification_type):   
    sgd = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # a stopping function should the validation loss stop improving
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    
    if classification_type == 'type':
        loss = 'categorical_crossentropy'
    
    else:
        loss = 'binary_crossentropy'

    model.compile(loss=loss, metrics=['accuracy'], optimizer=sgd)
    tensorboardRNN = TensorBoard(log_dir="RNN_logs/{}".format(time()))
    
    #for i in range(number_epoch):
    history1 = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), callbacks=[tensorboardRNN, earlystop], batch_size=32, epochs=int(number_epoch), shuffle=False)
    #model.reset_states()        
    
    print(model.summary())

    return model, history1


# In[8]:


def plotLoses(history1, number_opoch): 
    # visualizing losses and accuracy 
    train_loss = history1.history['loss']
    val_loss   = history1.history['val_loss']
    train_acc  = history1.history['acc']
    val_acc    = history1.history['val_acc']
    xc         = range(number_opoch)
   
    plt.plot(xc, train_loss, 'r--', label='Traing loss per epoch', linewidth=1.0)
    plt.title('Training and validation loss per epoch')
    #plt.xlabel('Epoch')
    #plt.ylabel('Traing loss')
    #plt.legend(loc='upper right')
    #plt.grid(True)
    
    plt.plot(xc, val_loss, 'c--', label='Validation loss per epoch', linewidth=1.0)
    #plt.title('Validation loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    #plt.grid(True)
    plt.show()
    
    plt.plot(xc, train_acc, 'r--', label='Training accuracy per epoch', linewidth=1.0)
    plt.title('Training vs validation accuracy per epoch')
    #plt.xlabel('Epoch')
    #plt.ylabel('Traing loss')
    #plt.legend(loc='upper right')
    #plt.grid(True)
    
    plt.plot(xc, val_acc, 'c--', label='Validation accuracy per epoch', linewidth=1.0)
    #plt.title('Validation loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    #plt.grid(True)


# In[9]:


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
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
        
        print ("F-Score:", round(f,3))
        print ("Precision:", round(p,3))
        print ("Recall:", round(r,3))
        print ("F-Score:", round(f,3)) 
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        
        class_names = ["COAD", "GBM", "KIRC", "LGG", "LUAD", "LUSC", "OV", "UCEC", "BRCA", "HNSC","THCA", "PRAD", "STAD", "BLCA"]
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix: true vs predicted label')  
        plt.show() 
    
        _test_data_class(y_true, y_pred, class_names)
        import gc; gc.collect()
    
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    else:
        auc = metrics.roc_auc_score(y_true, y_pred)  # Receiver operating characteristic curve a.k.a ROC is a graphic plot illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.      
        # 
        mcc = matthews_corrcoef(y_true, y_pred)
        
        print ("Area under ROC Curve:", round(auc,3))
        print ("Matthews correlation coefficient:", round(mcc,3)) # +1 means perfect prediction
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        
        class_names = ["Tumor", "Healthy"]

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix: true vs predicted label')  
        plt.show() 
    
        _test_data_class(y_true, y_pred, class_names)
        import gc; gc.collect()    

        print(classification_report(y_true, y_pred, target_names=class_names))


# In[17]:


type_features, type_labels = prepareData('type')
risk_features, risk_labels = prepareData('identification')


# In[11]:


def cancerClassifier(classification_type, number_epoch, features, labels, reg):

    #features, labels = prepareData(classification_type)

    labels = one_hot_encode(labels)
    train_x, test_x, train_y, test_y, valid_x, valid_y = prepare_test_train_valid(features, labels)

    data_dim = 569
    print(data_dim)

    timesteps = 1
    num_classes = train_y.shape[1]
    print(num_classes)

    model = model_build(num_classes, timesteps, data_dim, reg)

    trainedModel, history1 = model_train(model, number_epoch, train_x, train_y, valid_x, valid_y, classification_type)
    plotLoses(history1, number_epoch)
    model_evaluate(trainedModel, classification_type, test_x, test_y) 
    
    #return trainedModel, model

reg = L1L2(l1=0.01, l2=0.00) #Fake
cancerClassifier('type', 6, type_features, type_labels, reg)
