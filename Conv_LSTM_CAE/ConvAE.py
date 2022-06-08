# coding: utf-8
import keras
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop, SGD, Adam
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
from keras.optimizers import RMSprop
from keras.regularizers import l2

from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering("tf")
import matplotlib.pyplot as plt
import itertools
from keras.regularizers import L1L2
import numpy as np
import pandas as pd

np.random.seed(10)
from time import time
import numpy as np
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics

import keras.layers.normalization as bn
from sklearn.metrics import confusion_matrix
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
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Convolution2D,
    UpSampling1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
)
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Dense
from keras.layers import Flatten, Dense, Reshape
from keras.layers import LSTM
from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D, RepeatVector
from keras.models import Model
import pandas


def prepareData(classification_type):
    df = pd.read_csv("all_gene_input_data.csv", sep="\t")

    df = df.drop(df.columns[[0]], axis=1)

    if classification_type == "type":
        df = df[df["ctype"] != 15]
        for val in range(1, np.unique(df["ctype"]).size + 1):
            df.ix[df["ctype"] == val, "ctype"] = val - 1  # labels needs to start from 0

    if classification_type == "identification":
        df.ix[df["ctype"] != 15, "ctype"] = 0
        df.ix[df["ctype"] == 15, "ctype"] = 1

    column_headers = df.columns.values.tolist()
    column_headers.remove("ctype")

    features = df[column_headers].values
    labels = df["ctype"].values

    normalizedFeatures = normalize(
        features.reshape(features.shape[0], -1), norm="max", axis=0
    ).reshape(features.shape)

    print(type(normalizedFeatures))
    print(type(labels))

    # features = binary[cols]
    # labels = binary['ctype']

    features = np.array(normalizedFeatures)
    labels = np.array(labels)

    print(features.shape)
    print(labels.shape)

    # print(images.shape)
    features, labels = shuffle(features, labels, random_state=0)  # shuffle the data
    print("Shauffle completed!")

    return features, labels


def prepare_test_train_valid(features, labels):
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.25, random_state=12345
    )

    print("X_train shape:", train_x.shape)
    print("Y_train shape:", train_y.shape)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    return train_x, test_x, train_y, test_y


type_features, type_labels = prepareData("type")
type_features = np.reshape(
    type_features, (type_features.shape[0], 1, type_features.shape[1])
)
type_features = type_features.transpose(0, 2, 1)

print(type_features.shape)
print(type_labels.shape)

# Shapes of training set
print("Feature set shape: {shape}".format(shape=type_features.shape))

# Shapes of test set
print("Label array shape: {shape}".format(shape=type_labels.shape))

input_layer = Input(shape=(20308, 1))
num_classes = 14

# Let's create separate encoder and decoder functions since you will be using encoder weights later on for classification purpose!

# # Classification model


def encoder(input_layer):
    conv1 = Conv1D(32, 2, activation="relu", padding="same")(
        input_layer
    )  # 28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(32, 2, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=1, strides=2)(conv1)  # 14 x 14 x 32
    conv2 = Conv1D(64, 2, activation="relu", padding="same")(pool1)  # 14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(64, 2, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=1, strides=2)(conv2)  # 7 x 7 x 64
    conv3 = Conv1D(128, 2, activation="relu", padding="same")(
        pool2
    )  # 7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(128, 2, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(256, 2, activation="relu", padding="same")(
        conv3
    )  # 7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(256, 2, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)

    return conv4


def decoder(conv4):
    conv5 = Conv1D(128, 2, activation="relu", padding="same")(conv4)  # 7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(128, 2, activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(64, 2, activation="relu", padding="same")(conv5)  # 7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(64, 2, activation="relu", padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling1D(2)(conv6)  # 14 x 14 x 64
    conv7 = Conv1D(32, 2, activation="relu", padding="same")(up1)  # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(32, 2, activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling1D(2)(conv7)  # 28 x 28 x 32
    decoded = Conv1D(1, 2, activation="sigmoid", padding="same")(up2)  # 28 x 28 x 1

    return decoded


autoencoder = Model(input_layer, decoder(encoder(input_layer)))
autoencoder.compile(loss="mean_squared_error", optimizer=RMSprop())
autoencoder.summary()

autoencoder_train = autoencoder.fit(
    type_features,
    type_features,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_split=0.1,
)

loss = autoencoder_train.history["loss"]
val_loss = autoencoder_train.history["val_loss"]


def encoder(input_layer):
    conv1 = Conv1D(32, 2, activation="relu", padding="same")(
        input_layer
    )  # 28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(32, 2, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=1, strides=2)(conv1)  # 14 x 14 x 32
    conv2 = Conv1D(64, 2, activation="relu", padding="same")(pool1)  # 14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(64, 2, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=1, strides=2)(conv2)  # 7 x 7 x 64
    conv3 = Conv1D(128, 2, activation="relu", padding="same")(
        pool2
    )  # 7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(128, 2, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(256, 2, activation="relu", padding="same")(
        conv3
    )  # 7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(256, 2, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)

    return conv4


# Let's define the fully connected layers that you will be stacking up with the encoder function.


def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation="relu")(flat)
    out = Dense(num_classes, activation="softmax")(den)
    return out


encode = encoder(input_layer)
full_model = Model(input_layer, fc(encode))

for l1, l2 in zip(full_model.layers[:19], autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())

# Let's print first layer weights of both the models.

full_model.get_weights()[0][1]

# Voila! Both the arrays look exactly similar. So, without any further ado, let's compile the model and start the training. Next, you will make the encoder part i.e.the first nineteen layers of the model trainable false. Since the encoder part is already trained, you do not need to train it. You will only be training the Fully Connected part.

for layer in full_model.layers[0:19]:
    layer.trainable = False

full_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)
full_model.summary()

type_labels_one_hot = to_categorical(type_labels)

# Display the change for category label using one-hot encoding
print("Original label:", type_labels[0])
print("After conversion to one-hot:", type_labels_one_hot[0])

from sklearn.utils import class_weight

train_x, test_x, train_y, test_y = prepare_test_train_valid(
    type_features, type_labels_one_hot
)

sample_weights = class_weight.compute_sample_weight("balanced", train_y)
print(train_x.shape)
train_x = train_x.transpose(0, 2, 1)
test_x = test_x.transpose(0, 2, 1)

print(train_x.shape)
print(test_y.shape)

# # Train the Model

classify_train = full_model.fit(
    train_x, train_y, validation_split=0.1, batch_size=64, epochs=10, verbose=1
)
full_model.save_weights("autoencoder_classification.h5")

# Next, you will re-train the model by making the first nineteen layers trainable as True instead of keeping them False! So, let's quickly do that.

for layer in full_model.layers[0:19]:
    layer.trainable = True

full_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)
classify_train = full_model.fit(
    train_x, train_y, batch_size=64, epochs=10, verbose=1, validation_split=0.1
)
full_model.save_weights("classification_complete.h5")

accuracy = classify_train.history["acc"]
val_accuracy = classify_train.history["val_acc"]
loss = classify_train.history["loss"]
val_loss = classify_train.history["val_loss"]

# # Model Evaluation on the Test Set
test_eval = full_model.evaluate(test_x, test_y, verbose=0)
print("Test loss:", test_eval[0])
print("Test accuracy:", test_eval[1])

predicted_classes = full_model.predict(test_x)
pred = np.argmax(np.round(predicted_classes), axis=1)
y = np.argmax(np.round(test_y), axis=1)

print(y[0])
print(pred[0])

from sklearn.metrics import classification_report

# target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y, pred))
