"""
@author: Brendan Bassett
@date: 05/09/2022

CS 3120 MACHINE LEARNING
MSU Denver
Dr. Feng Jiang

------------------------------------------------------------------------------------------------------------------------

FINAL PROJECT - Optical Music Character Recognition

Categorizes individual music symbols by type using 3 separate machine learning strategies. Data is processed through
each, then each result becomes a vote towards the final determination.

Process 1: Convolutional Neural Network
Process 2: Convolutional Neural Network
Process 3: Logistic Regression

------------------------------------------------------------------------------------------------------------------------

DATASET SOURCE

Sources:        http://www.inescporto.pt/~jsc/projects/OMR/
                https://apacha.github.io/OMR-Datasets/#rebelo-dataset

License:        CC-BY-SA

Author:         A. Rebelo, G. Capela, J. S. Cardoso

Publication:    “Optical recognition of music symbols: A comparative study”
                International Journal on Document Analysis and Recognition, vol. 13, no. 1, pp. 19-31, 2010.
                DOI: 10.1007/s10032-009-0100-1
"""


# SETUP ****************************************************************************************************************

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import scikitplot as skplt
import tensorflow
from sklearn import linear_model
from sklearn import metrics
from numpy.core.records import ndarray
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential


# CONSTANTS ************************************************************************************************************

ROOT_PATH = os.path.realpath(os.path.dirname(__file__))

IMG_SHAPE = (20, 20, 3)
CNN1_EPOCHS = 4
CNN2_EPOCHS = 5

TRAIN_SIZE = 0.6
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.1

LOG_REG_MAX_ITER = 2000

NUM_CLASSES = 15
LABELS = ['Accent', 'AltoCleff', 'BassClef', 'Breve', 'Flat', 'Naturals', 'Notes', 'NotesFlags', 'NotesOpen',
          'Rests1', 'Rests2', 'Sharps', 'TimeSignatureL', 'TimeSignatureN', 'TrebleClef']


# FUNCTIONS ************************************************************************************************************


def cnn1():

    print("\n-----------------------------------------------")
    print("-------  Convolution Neural Network 1  ----------")
    print("-----------------------------------------------\n")

    # # LOAD the preprocessed data from a file so the data is the same for each model.
    train_x, train_y, test_x, test_y, val_x, val_y = load_test_train_val(True)

    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(24, kernel_size=(5, 5), padding="same", activation='relu', input_shape=IMG_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # model.add(Dense(num_classes, activation='softmax'))

    print("test_x.shape: ", test_x.shape)
    print("test_y.shape: ", test_y.shape)

    # Train the model using Stochastic Gradient Descent.
    sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=CNN2_EPOCHS)

    # Assess the performance of the neural network.
    pred_y = model.predict(test_x)
    print(classification_report(test_y.argmax(axis=1), pred_y.argmax(axis=1)))

    # plot the training loss and accuracy
    plt.figure()
    plt.style.use("ggplot")
    plt.plot(np.arange(0, CNN2_EPOCHS), history.history["loss"], label="Training Loss")
    plt.plot(np.arange(0, CNN2_EPOCHS), history.history["val_loss"], label="Testing Loss")
    plt.plot(np.arange(0, CNN2_EPOCHS), history.history["accuracy"], label="Training Accuracy")
    plt.plot(np.arange(0, CNN2_EPOCHS), history.history["val_accuracy"], label="Testing Accuracy")
    plt.title("CNN1: Training and Testing Loss & Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

    return test_x, test_y, pred_y.argmax(axis=1)


def cnn2():

    print("\n-----------------------------------------------")
    print("--------  Convolution Neural Network 2 ----------")
    print("-----------------------------------------------\n")

    # This is a very simple convolution-based neural network. It is the most accurate, and it is the fastest to run of
    # the three models used in this project.

    # # LOAD the preprocessed data to a file, so we can skip preprocessing when testing the neural network.
    train_x, train_y, test_x, test_y, val_x, val_y = load_test_train_val(True)

    model = Sequential()
    model.add(Conv2D(26, kernel_size=(7, 7), padding="same", activation='relu', input_shape=IMG_SHAPE))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # model.add(Dense(num_classes, activation='softmax'))

    print("test_x.shape: ", test_x.shape)
    print("test_y.shape: ", test_y.shape)

    # Train the model using Stochastic Gradient Descent.
    sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=CNN1_EPOCHS)

    # Assess the performance of the neural network.
    pred_y = model.predict(test_x)
    print(classification_report(test_y.argmax(axis=1), pred_y.argmax(axis=1)))

    # Plot the training loss and accuracy.
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, CNN1_EPOCHS), history.history["loss"], label="Training Loss")
    plt.plot(np.arange(0, CNN1_EPOCHS), history.history["val_loss"], label="Testing Loss")
    plt.plot(np.arange(0, CNN1_EPOCHS), history.history["accuracy"], label="Training Accuracy")
    plt.plot(np.arange(0, CNN1_EPOCHS), history.history["val_accuracy"], label="Testing Accuracy")
    plt.title("CNN 2: Training and Testing Loss & Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

    return test_x, test_y, pred_y.argmax(axis=1)


def log_reg():

    print("\n-----------------------------------------------")
    print("---------   Logistic Classification  ----------")
    print("-----------------------------------------------\n")

    # # LOAD the preprocessed data to a file, so we can skip preprocessing when testing the neural network.
    train_x, train_y, test_x, test_y, val_x, val_y = load_test_train_val(False)

    # Convert labels to integers.
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(LABELS)

    # Fit the model with training data and test it after fitting.
    # X data must be standardized so logistic regression can complete in < 1000 iterations.
    regression = linear_model.LogisticRegression(max_iter=LOG_REG_MAX_ITER)
    regression.fit(train_x, train_y)
    pred_y = regression.predict(test_x)

    print(metrics.classification_report(test_y, pred_y, target_names=label_encoder.classes_))

    plt.figure()
    skplt.metrics.plot_confusion_matrix(test_y, pred_y, title=("Logistic Regression Confusion Matrix"), cmap="Reds")
    plt.yticks(label_encoder.fit_transform(label_encoder.classes_), label_encoder.classes_)
    plt.show()

    return test_x, test_y, pred_y


def load_from_images():
    #   Get the path for the dataset from wherever the user has placed it on their machine.
    #       Assume project file structures are unchanged.
    data_path = os.path.join(ROOT_PATH + '\\Rebelo_dataset')

    print("Data from: ", data_path)

    # Load the data and their labels into a single data frame. Labels are the folder name for each symbol type.
    data = []
    labels = []
    class_folders = os.listdir(data_path)
    print(class_folders)

    for class_name in class_folders:
        image_list = os.listdir(os.path.join(data_path + '\\' + str(class_name)))

        for image_name in image_list:
            img_path = os.path.join(data_path + '\\' + str(class_name) + '\\' + image_name)
            img = cv.imread(img_path)
            # img = bound_image(img)
            data.append(img)
            labels.append(class_name)

    data = np.array(data)
    labels = np.array(labels)

    # Encode the labels as integers
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # show some information on memory consumption of the images
    print("\nData size: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

    return data, labels


def load_from_numpy(file_name: str):
    file_path = os.path.join(ROOT_PATH + "\\Numpy_Save\\" + file_name)

    if not os.path.exists(file_path):
        print("No Numpy_Save directory! Please put the numpy file in directory:   ", file_path)
        return

    return np.load(file_path)


def load_test_train_val(cnn: bool):
    if cnn:
        train_x = load_from_numpy("cnn_train_x.npy")
        train_y = load_from_numpy("cnn_train_y.npy")
        test_x = load_from_numpy("cnn_test_x.npy")
        test_y = load_from_numpy("cnn_test_y.npy")
        val_x = load_from_numpy("cnn_val_x.npy")
        val_y = load_from_numpy("cnn_val_y.npy")

    else:
        train_x = load_from_numpy("lr_train_x.npy")
        train_y = load_from_numpy("lr_train_y.npy")
        test_x = load_from_numpy("lr_test_x.npy")
        test_y = load_from_numpy("lr_test_y.npy")
        val_x = load_from_numpy("lr_val_x.npy")
        val_y = load_from_numpy("lr_val_y.npy")

    return train_x, train_y, test_x, test_y, val_x, val_y


def save_test_train_val(train_x, train_y, test_x, test_y, val_x, val_y, cnn: bool):
    if cnn:
        save_to_numpy(train_x, "cnn_train_x.npy")
        save_to_numpy(train_y, "cnn_train_y.npy")
        save_to_numpy(test_x, "cnn_test_x.npy")
        save_to_numpy(test_y, "cnn_test_y.npy")
        save_to_numpy(val_x, "cnn_val_x.npy")
        save_to_numpy(val_y, "cnn_val_y.npy")

    else:
        save_to_numpy(train_x, "lr_train_x.npy")
        save_to_numpy(train_y, "lr_train_y.npy")
        save_to_numpy(test_x, "lr_test_x.npy")
        save_to_numpy(test_y, "lr_test_y.npy")
        save_to_numpy(val_x, "lr_val_x.npy")
        save_to_numpy(val_y, "lr_val_y.npy")


def save_to_numpy(data: ndarray, file_name: str):
    numpy_save_dir = os.path.join(ROOT_PATH + "\\Numpy_Save\\")

    if not os.path.exists(numpy_save_dir):
        os.mkdir(numpy_save_dir)
        print("No Numpy_Save directory. Created new one:  ", numpy_save_dir)

    file_path = os.path.join(numpy_save_dir + file_name)
    np.save(file_path, data)


def test_train_val_split(data, labels, train_per: float, test_per: float, val_per: float):

    if train_per + test_per + val_per != 100.:
        print("ERROR:: test_train_val_split percentages must add up to 100%.")
        print("    train_per: ", train_per)
        print("    test_per: ", test_per)
        print("    val_per: ", val_per)

    # Split the data into training, testing, and validation.
    test_val_size = (100. - train_per) / 100.
    train_x, rem_x, train_y, rem_y = train_test_split(data, labels, test_size=test_val_size, random_state=42)

    val_size = (val_per / 100.) / test_val_size
    test_x, val_x, test_y, val_y = train_test_split(rem_x, rem_y, test_size=val_size, random_state=42)

    # print("Training % ::", round(float(len(train_x)) / len(data) * 100, 2))
    # print("Testing % ::", round(float(len(test_x)) / len(data) * 100, 2))
    # print("Validation % ::", round(float(len(val_x)) / len(data) * 100, 2))

    print("\nTraining size ::", str(len(train_x)))
    print("Testing size :: ", str(len(test_x)))
    print("Validation size ::", str(len(val_x)))

    return train_x, train_y, test_x, test_y, val_x, val_y


def vote(test_x, test_y, cnn1_pred_y, cnn2_pred_y, log_reg_pred_y):

    pred_final = np.zeros(len(test_y))

    for i, x in enumerate(test_x):
        vote_bin = np.zeros(15)
        vote_bin[cnn1_pred_y[i]] += 1
        vote_bin[cnn2_pred_y[i]] += 1
        vote_bin[log_reg_pred_y[i]] += 1

        vote_tie = True
        for j, cat in enumerate(vote_bin):
            if cat == 3:
                # This category got all 3 votes
                pred_final[i] = j
                vote_tie = False
                break

            elif cat == 2:
                # This category got the most of 3 votes.
                pred_final[i] = j
                vote_tie = False

                print("\nThis data point did not get all the votes")
                print("i:", i)
                print("cnn1_pred_y: ", cnn1_pred_y[i])
                print("cnn2_pred_y: ", cnn2_pred_y[i])
                print("log_reg_pred_y: ", log_reg_pred_y[i])
                print("pred_final: ", pred_final[i])
                print("test_y: ", test_y[i])
                break

        if vote_tie:
            # If 3-way tie, default to cnn2 results since it has the best stand-alone performance
            pred_final[i] = cnn2_pred_y[i]

            print("\nThis data point had a 3-way tie between models")
            print("i:", i)
            print("cnn1_pred_y: ", cnn1_pred_y[i])
            print("cnn2_pred_y: ", cnn2_pred_y[i])
            print("log_reg_pred_y: ", log_reg_pred_y[i])
            print("pred_final: ", pred_final[i])
            print("test_y: ", test_y[i])

    return pred_final


def main():

    print("\n---- CNN PREPROCESSING ----\n")

    data, labels = load_from_images()
    train_x, train_y, test_x, test_y, val_x, val_y = test_train_val_split(data, labels, 60., 30., 10.)

    # Invert binary images for masking. White <--> Black
    train_x = np.invert(train_x)
    test_x = np.invert(test_x)
    val_x = np.invert(val_x)

    # Flatten the data.
    train_x = train_x.reshape(train_x.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 3).astype('float') / 255.0
    test_x = test_x.reshape(test_x.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 3).astype('float') / 255.0
    val_x = val_x.reshape(val_x.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 3).astype('float') / 255.0

    # Convert labels to vectors.
    train_y = keras_utils.to_categorical(train_y, num_classes=NUM_CLASSES)
    test_y = keras_utils.to_categorical(test_y, num_classes=NUM_CLASSES)
    val_y = keras_utils.to_categorical(val_y, num_classes=NUM_CLASSES)

    # SAVE the preprocessed CNN data to a file. This way we can ensure train, test, val datasets are equivalent
    # and we can skip preprocessing when testing the neural networks.
    save_test_train_val(train_x, train_y, test_x, test_y, val_x, val_y, True)

    print("\n---- LOG REG PREPROCESSING ----\n")
    data = data.reshape((data.shape[0], IMG_SHAPE[0] * IMG_SHAPE[1] * 3))
    train_x, train_y, test_x, test_y, val_x, val_y = test_train_val_split(data, labels, 60., 30., 10.)

    # Invert binary images for masking. White <--> Black
    train_x = np.invert(train_x)
    test_x = np.invert(test_x)
    val_x = np.invert(val_x)

    save_test_train_val(train_x, train_y, test_x, test_y, val_x, val_y, False)

    print("\n---- MODELS ----\n")

    # Run the models. test_x and test_y

    test_x, test_y, pred_y_cnn1 = cnn1()
    test_x, test_y, pred_y_cnn2 = cnn2()
    test_x, test_y, pred_y_logreg = log_reg()

    pred_final = vote(test_x, test_y, pred_y_cnn1, pred_y_cnn2, pred_y_logreg)

    # Convert labels to integers.
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(LABELS)
    print(metrics.classification_report(test_y, pred_final, target_names=label_encoder.classes_))

    plt.figure()
    skplt.metrics.plot_confusion_matrix(test_y, pred_final, title="FINAL RESULTS CONFUSION MATRIX", cmap="Reds")
    plt.yticks(label_encoder.fit_transform(label_encoder.classes_), label_encoder.classes_)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# This function is not currently necessary, but it may come in handy later.
# def bound_image(img: ndarray):
#
#     # This method inverts each input image and scales them to the same size by adding black borders if need be.
#
#     # Invert the image for masking. Black <--> White
#     inv_img = cv.bitwise_not(img)
#
#     # If no scaling needs to be done, return the inverted image.
#     if img.shape == IMG_SHAPE:
#         return inv_img
#
#     # Ensure that we are scaling up in size, not down. If we try to scale down on accident this error will occur.
#     if img.shape[0] > IMG_SHAPE[0] or img.shape[1] > IMG_SHAPE[1]:
#         print("ERROR: Original image is larger than the desired image size. Cannot convert down in size.")
#         cv.imshow("ERROR: OG Image too large for conversion", img)
#         cv.waitKey(0)
#         return
#
#     # Resize the image so they are uniform by adding black borders around the original image.
#     delta_x = IMG_SHAPE[0] - inv_img.shape[0]
#     delta_y = IMG_SHAPE[1] - inv_img.shape[1]
#
#     # Split the extra pixels exactly between each border to ensure desired shape.
#     t_border = delta_y // 2
#     b_border = delta_y // 2
#     l_border = delta_x // 2
#     r_border = delta_x // 2
#
#     if (delta_x % 2) == 1:
#         l_border += 1
#     if (delta_y % 2) == 1:
#         t_border += 1
#
#     new_img = cv.copyMakeBorder(inv_img, t_border, b_border, l_border, r_border, cv.BORDER_CONSTANT, value=[0, 0, 0])
#
#     if new_img.shape[0] != IMG_SHAPE[0] or new_img.shape[1] != IMG_SHAPE[1]:
#         print("ERROR: The image did not resize to the new shape for some reason.  new_img.shape: ", new_img.shape)
#
#     # cv.imshow("Original Image", img)
#     # cv.waitKey(0)
#     # cv.imshow("Inverted, Resized Image", new_img)
#     # cv.waitKey(0)
#
#     return new_img
