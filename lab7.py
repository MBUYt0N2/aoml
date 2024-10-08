import os
import random
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import l1, l2
import numpy as np
from sklearn.model_selection import train_test_split

EPOCHS = 5
os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = 10

    # Flatten and normalize the data
    x_train = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255
    x_test = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, num_classes


def create_optimizers():

    # TODO: modify the below statements to return the corresponding optimizers in the dictionary
    # For SGD, set the learning rate as 0.01. For the rest, set the learning rate as 0.001.
    # For SGD, set momentum as 0.9

    # return {
    #     'SGD':
    #     'Adam':
    #     'RMSprop':
    # }

    d = {}
    d["SGD"] = SGD(learning_rate=0.01, momentum=0.9)
    d["Adam"] = Adam(learning_rate=0.001)
    d["RMSprop"] = RMSprop(learning_rate=0.001)
    return d


def create_regularizers():
    # TODO: modify the below statements to return the corresponding regularizers in the dictionary
    # For all models, set the penalty as 10^-5

    # return {
    #     'None': None,
    #     'L1':
    #     'L2':
    # }

    d = {}
    d["None"] = None
    d["L1"] = tf.keras.regularizers.l1(1e-5)
    d["L2"] = tf.keras.regularizers.l2(1e-5)
    return d


def create_model(optimizer, regularization=None, num_classes=10):
    tf.random.set_seed(42)

    # TODO: Create a model with the following layers:
    # 1. Dense: 2048 neurons, relu activation
    # 2. Dense: 1024 neurons, relu activation
    # 3. Dense: 512 neurons, relu activation
    # 4. Dense: 10 neurons, softmax activation
    # Layers 1, 2 and 3 must have the regularization applied on their activation
    # Finally, compile model with the optimizer (already done)
    model = tf.keras.Sequential(
        [
            Dense(2048, activation="relu", kernel_regularizer=regularization),
            Dense(1024, activation="relu", kernel_regularizer=regularization),
            Dense(512, activation="relu", kernel_regularizer=regularization),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


def train_and_evaluate():
    x_train, y_train, x_test, y_test, num_classes = load_data()

    # Split training data to create a validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    results = {}

    for reg_name, regularization in create_regularizers().items():
        for opt_name, optimizer in create_optimizers().items():
            print(f"\nTraining with {opt_name} optimizer and {reg_name} regularization")

            model = create_model(optimizer, regularization, num_classes)

            history = model.fit(
                x_train,
                y_train,
                batch_size=128,
                epochs=EPOCHS,
                verbose=1,
                validation_data=(x_val, y_val),
            )

            test_score = model.evaluate(x_test, y_test, verbose=0)
            results[f"{opt_name}_{reg_name}"] = {
                "test_loss": test_score[0],
                "test_accuracy": test_score[1],
                "history": history.history,
            }

    return results


if __name__ == "__main__":
    results = train_and_evaluate()

    # Print final results
    print("\nFinal Results:")
    for config, result in results.items():
        print(f"{config}:")
        print(f"  Test accuracy: {result['test_accuracy']:.4f}")
        print(f"  Test loss: {result['test_loss']:.4f}")
