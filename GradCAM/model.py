from keras_preprocessing.image import ImageDataGenerator

INPUT_SHAPE = (144, 144)
N_CLASES = 5
L2_REG = 0.0001

import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Activation,
    Conv2D,
    BatchNormalization,
    Dropout,
    MaxPool2D,
    Flatten
)


def build_model():
    model = Sequential(
        [
            Conv2D(
                64,
                kernel_size=5,
                strides=1,
                padding="same",
                input_shape=INPUT_SHAPE + (1,),
            ),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.25),
            Conv2D(128, kernel_size=5, strides=1, padding="same"),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.25),
            Conv2D(256, kernel_size=3, strides=1, padding="same"),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),
            Activation("relu", name="last_conv"),
            Dropout(0.25),
            Flatten(),
            Dense(512),
            Activation("relu"),
            Dense(512),
            Dense(N_CLASES),
            Activation("softmax"),
        ]
    )
    return model


def load_model(k_fold):
    model = build_model()
    model.load_weights(model_path(k_fold))
    return model


def model_path(k_fold):
    return "saved_models/model_fold" + str(k_fold) + ".h5"


def train(k_fold):
    cross_validation_path = "data/crossValidation/" + str(k_fold)
    model = build_model()
    opt = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=cross_validation_path + "/train/",
        target_size=INPUT_SHAPE,
        color_mode="grayscale",
        batch_size=24,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

    valid_generator = valid_datagen.flow_from_directory(
        directory=cross_validation_path + "/dev",
        target_size=INPUT_SHAPE,
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        seed=42,
    )

    path = model_path(k_fold)
    model.load_weights(path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        period=1,
    )
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=20,
        validation_data=valid_generator,
        validation_steps=7,
        epochs=50,
        verbose=1,
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    for fold in range(0, 5):
        train(fold)
