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
import math

import cv2 as cv
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import os
import scikitplot as skplt
import tensorflow
import keras.utils as keras_utils
from sklearn import linear_model
from sklearn import metrics
from numpy.core.records import ndarray
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

# ============== CONSTANTS =========================================================================================

ROOT_PATH = os.path.realpath(os.path.dirname(__file__))
ANN_IMG_PATH = os.path.join("F://OMR_Datasets/DS2_Transformed/annotations/")

SQL_DENSE = 'ds2_dense'
SQL_DENSE_TEST = 'ds2_dense_test'
SQL_DENSE_TRAIN = 'ds2_dense_train'

ANNOTATION_SET = 'deepscores'  # Annotation category set. Can be 'deepscores' or 'muscima++'

CNN1_EPOCHS = 4
CNN2_EPOCHS = 5

TRAIN_SIZE = 0.6
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.1

LOG_REG_MAX_ITER = 2000


# ============== VARIABLES =========================================================================================


# ============== FUNCTIONS =========================================================================================

# Convert all the pixelwise annotations for a database into numpy data arrays and labels.
def convert_sql_to_numpy(sql_src: str):
    # Clear the instance numpy datasets in case they already have something in them.
    categories = {}
    ann_meta = np.array([])
    ann_imgs = np.array([])
    ann_ids = np.array([])  # The annotation ids in the order that they appear in our numpy dataset.
    ann_cat_ids = np.array([])  # Y-values. The correct category for each annotation.

    print("\n\n***  Initializing MySQL %s DB connection...  ***\n" % sql_src)

    try:
        connection = mysql.connector.connect(user='root', password='MusicE74!', host='localhost',
                                             db=sql_src, buffered=True)
        cursor = connection.cursor()

        # Load the list of possible categories for the chosen annotation set.
        print("\nLoading categories...")

        message = "SELECT id, `name` FROM categories WHERE annotation_set = \'%s\'" % ANNOTATION_SET
        cursor.execute(message)
        connection.commit()
        fetch = cursor.fetchall()

        for row in fetch:
            categories[row[0]] = row[1]

        labels = np.array(list(categories.keys()))

        # Load the appropriate category for each annotation.
        print("\nLoading annotation_categories...")

        message = "SELECT ann_id, cat_id FROM annotations_categories INNER JOIN categories cat " \
                  "ON (cat.id = cat_id AND cat.annotation_set=\'%s\')" % ANNOTATION_SET
        cursor.execute(message)
        connection.commit()
        fetch = cursor.fetchall()

        # Convert to dict for hashing during merge with annotation data.
        ann_cats = {}
        for row in fetch:
            ann_cats[row[0]] = row[1]  # ann_cat.ann_id (key), ann_cat.cat_id (value)

        # Obtain a list of every source image referred to by this database.into
        print("\nLoading images list...")
        message = "SELECT id FROM images"
        cursor.execute(message)
        connection.commit()
        fetch = cursor.fetchall()

        img_ids = []
        for row in fetch:
            img_ids.append(row[0])
        print("\n   img_ids:", img_ids)

        # Load the annotation data from the SQL database.
        # Load the pixelwise annotation data from jpg files.
        for img_id in img_ids:
            print("\nLoading annotations from image", img_id)

            message = "SELECT file_name FROM images WHERE id = \'%s\'" % img_id
            cursor.execute(message)
            connection.commit()
            file_name = cursor.fetchall()[0][0]  # [0][0] Otherwise returns a one-entry array with a one-entry tuple.

            # Get the annotation data most useful for our model. Only get the annotations for a single image at a time.
            # Add the annotation's category which corresponds to the annotation set being used in the model.
            message = "SELECT ann.id, ann_cat.cat_id, " \
                      "a_bbox_x0, a_bbox_y0, a_bbox_x1, a_bbox_y1, " \
                      "o_bbox_x0, o_bbox_y0, o_bbox_x1, o_bbox_y1, " \
                      "o_bbox_x2, o_bbox_y2, o_bbox_x3, o_bbox_y3 FROM annotations ann " \
                      "INNER JOIN annotations_categories ann_cat INNER JOIN categories cat ON (ann.img_id=\'%s\' " \
                      "AND ann.id=ann_cat.ann_id AND ann_cat.cat_id=cat.id AND cat.annotation_set=\'%s\')" \
                      % (img_id, ANNOTATION_SET)
            cursor.execute(message)
            connection.commit()
            fetch = cursor.fetchall()

            # Load & process the data for every annotation on this image.
            for row in fetch:

                id, cat_id, meta, img = extract_single_annotation(row)

                if img is None:     # The image file may have failed to load.
                    continue

                ann_ids = np.append(ann_ids, id)
                ann_cat_ids = np.append(ann_cat_ids, cat_id)
                ann_meta = np.append(ann_meta, meta)
                ann_imgs = np.append(ann_imgs, img)

        print("ann_imgs:", ann_imgs)
        print("\nData size: {:.1f}MB".format(ann_imgs.nbytes / (1024 * 1000.0)))

        # # Flatten the image data.
        # ann_imgs = ann_imgs.reshape(ann_imgs.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 3).astype('float') / 255.0
        # val_x = val_x.reshape(val_x.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 3).astype('float') / 255.0

    finally:
        print("\n\n***  MySQL Connection closed.  ***\n\n")
        connection.close()
        cv.destroyAllWindows()

    return labels, ann_ids, ann_cat_ids, ann_meta, ann_imgs


def cnn1(categories):
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
    model.add(Dense(len(categories), activation='softmax'))
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


def cnn2(categories):
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
    model.add(Dense(len(categories), activation='softmax'))
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


# Load a single annotation and its metadata from a row of SQL data into the instance numpy datasets.
def extract_single_annotation(row):
    id = row[0]
    cat_id = row[1]

    height = abs(row[4] - row[2])
    width = abs(row[5] - row[3])

    # Select the columns for o_bbox
    start_index = 5
    o_bbox = []
    for i in range(0, 4):
        o_bbox.append((row[start_index + i * 2], row[start_index + i * 2 + 1]))

    # Find the smallest magnitude angle from -pi/4 to pi/4. This represents the orientation of o_bbox.
    # Calculate the tan angle for each side. Watch the divide by zero errors.
    angles = [0, 0, 0, 0]

    for i in range(0, 4):
        x0 = o_bbox[i][0]
        y0 = o_bbox[i][1]

        if i == 3:
            x1 = o_bbox[0][0]  # Loop back to the first entry in the array.
            y1 = o_bbox[0][1]  # This avoids index out of bounds error.
        else:
            x1 = o_bbox[i + 1][0]
            y1 = o_bbox[i + 1][1]

        if y1 - y0 == 0:  # Avoid dividing by zero.
            angles[i] = 999  # We arbitrarily use 999 to represent infinity.
        elif x1 - x0 == 0:  # Avoid tan(0) where angle = 0.
            angles[i] = 0
        else:
            angles[i] = math.tan((x1 - x0) / (y1 - y0))

    orientation = math.pi / 4
    for i in range(0, 4):
        if abs(angles[i] < orientation):
            orientation = angles[i]

    # print("angles:", angles, "\n   o_bbox:", o_bbox, "\n   orientation :", orientation)

    img_path = ANN_IMG_PATH + str(id) + ".jpg"
    if not os.path.exists(img_path):
        print("\nThe annotation image does not exist! Skipped loading this image\n   img_path:   ", img_path)

    try:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    except cv.Error as e:
        print(e)

    if img is None:
        print(" !!! NO IMAGE WAS LOADED !!! \n id:", id)

    return id, cat_id, (height, width, orientation), img


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


def log_reg(categories):
    print("\n-----------------------------------------------")
    print("---------   Logistic Classification  ----------")
    print("-----------------------------------------------\n")

    # # LOAD the preprocessed data to a file, so we can skip preprocessing when testing the neural network.
    train_x, train_y, test_x, test_y, val_x, val_y = load_test_train_val(False)

    # Convert labels to integers.
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(categories)

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


# ============== MAIN CODE =========================================================================================

print("\n----  DATA PREPROCESSING  ----\n")

# Load the data
convert_sql_to_numpy(SQL_DENSE_TEST)
# convert_sql_to_numpy(SQL_DENSE_TRAIN)

# # Encode the labels as integers
# label_encoder = preprocessing.LabelEncoder()
# labels = label_encoder.fit_transform(labels)
#
# # Convert labels to vectors.
# train_y = keras_utils.to_categorical(train_y, num_classes=NUM_CLASSES)
# val_y = keras_utils.to_categorical(val_y, num_classes=NUM_CLASSES)
#
# # SAVE the preprocessed CNN data to a file. This way we can ensure train, test, val datasets are equivalent
# # and we can skip preprocessing when testing the neural networks.
# save_test_train_val(train_x, train_y, test_x, test_y, val_x, val_y, True)
#
# print("\n---- LOG REG PREPROCESSING ----\n")
# data = data.reshape((data.shape[0], IMG_SHAPE[0] * IMG_SHAPE[1] * 3))
# train_x, train_y, test_x, test_y, val_x, val_y = test_train_val_split(data, labels, 60., 30., 10.)
#
# # Invert binary images for masking. White <--> Black
# train_x = np.invert(train_x)
# test_x = np.invert(test_x)
# val_x = np.invert(val_x)
#
# save_test_train_val(train_x, train_y, test_x, test_y, val_x, val_y, False)
#
# print("\n---- MODELS ----\n")
#
# # Run the models. test_x and test_y
#
# test_x, test_y, pred_y_cnn1 = cnn1()
# test_x, test_y, pred_y_cnn2 = cnn2()
# test_x, test_y, pred_y_logreg = log_reg()
#
# pred_final = vote(test_x, test_y, pred_y_cnn1, pred_y_cnn2, pred_y_logreg)
#
# # Convert labels to integers.
# label_encoder = preprocessing.LabelEncoder()
# label_encoder.fit_transform(LABELS)
# print(metrics.classification_report(test_y, pred_final, target_names=label_encoder.classes_))
#
# plt.figure()
# skplt.metrics.plot_confusion_matrix(test_y, pred_final, title="FINAL RESULTS CONFUSION MATRIX", cmap="Reds")
# plt.yticks(label_encoder.fit_transform(label_encoder.classes_), label_encoder.classes_)
# plt.show()
#
#
# ============== LEGACY FUNCTIONS ======================================================================================
#
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
