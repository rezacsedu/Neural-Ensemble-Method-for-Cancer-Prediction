# coding: utf-8
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(10)


def prepareData(path):
    df = pd.read_csv(path, sep="\t")
    df = df.drop(df.columns[[0]], axis=1)

    df = df[df["ctype"] != 15]
    for val in range(1, np.unique(df["ctype"]).size + 1):
        df.ix[df["ctype"] == val, "ctype"] = val - 1  # labels needs to start from 0

    column_headers = df.columns.values.tolist()
    column_headers.remove("ctype")

    features = df[column_headers].values
    labels = df["ctype"].values

    normalizedFeatures = normalize(
        features.reshape(features.shape[0], -1), norm="max", axis=0
    ).reshape(features.shape)

    features = np.array(normalizedFeatures)
    labels = np.array(labels)

    print(features.shape)
    print(labels.shape)

    features, labels = shuffle(features, labels, random_state=0)  # shuffle the data
    print("Shauffle completed!")

    return features, labels


def prepare_test_train_valid(features, labels):
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.05, random_state=42
    )
    test_x, valid_x, test_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.20, random_state=42
    )

    print("X_train shape:", train_x.shape)
    print("Y_train shape:", train_y.shape)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    return train_x, test_x, train_y, test_y
