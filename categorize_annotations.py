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
import logging
import math
import threading

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
DS2_DENSE_SAVE_PATH = os.path.join("F://OMR_Datasets/DS2_Transformed")

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

# Convert all the pixelwise annotations for a database into numpy data arrays and labels. Then save them into flat
# files for later use.
def convert_sql_to_numpy(sql_src: str):
    # Clear the instance numpy datasets in case they already have something in them.
    categories = {}

    logging.info("***  Initializing MySQL %s DB connection...  ***" % sql_src)

    try:
        connection = mysql.connector.connect(user='root', password='MusicE74!', host='localhost',
                                             db=sql_src, buffered=True)
        cursor = connection.cursor()

        # Load the list of possible categories for the chosen annotation set.
        logging.info("Loading categories...")

        message = "SELECT id, `name` FROM categories WHERE annotation_set = \'%s\'" % ANNOTATION_SET
        cursor.execute(message)
        connection.commit()
        fetch = cursor.fetchall()

        for row in fetch:
            categories[row[0]] = row[1]

        labels = np.array(list(categories.keys()))

        # Load the appropriate category for each annotation.
        logging.info("Loading annotation_categories...")

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
        logging.info("Loading images list...")
        message = "SELECT id FROM images"
        cursor.execute(message)
        connection.commit()
        fetch = cursor.fetchall()

        img_ids = []
        for row in fetch:
            img_ids.append(row[0])
        logging.info("   img_ids: %s" % img_ids)

        ann_ids_all = []        # These Python lists each contain references to other lists, each containing annotation
        ann_cat_ids_all = []    # data for a single image. This way we can load the data quickly without copying
        ann_metas_all = []       # large numpy arrays. After loading we will append each sub-list to a numpy array.
        ann_imgs_all = []
        ann_fails_all = []

        # Load the annotation data from the SQL database.
        # Load the pixelwise annotation data from jpg files.
        for img_id in img_ids:
            ids, c_ids, metas, imgs, fails = load_single_image(img_id, connection, cursor)

            ann_ids_all.append(ids)  # The attach a REFERENCE to the data array we created. Later we will use the
            ann_cat_ids_all.append(c_ids)  # reference to merge into a numpy array. This process enables multithreading.
            ann_metas_all.append(metas)  # If we appended to numpy then numpy will copy the array for each append,
            ann_imgs_all.append(imgs)  # which is much too slow.
            ann_fails_all.append(fails)

        # Consolidate the annotation data into numpy arrays by appending them in groups (by image).

        ann_ids_all_np = np.array([])
        ann_cat_ids_all_np = np.array([])
        ann_metas_all_np = np.array([])
        ann_imgs_all_np = np.array([])
        ann_fails_all_np = np.array([])

        logging.info("Consolidating data into numpy arrays...")

        for i, ann_id in enumerate(ann_ids_all):
            ann_ids_all_np = np.append(ann_ids_all_np, ann_id)
            ann_cat_ids_all_np = np.append(ann_cat_ids_all_np, ann_cat_ids_all[i])
            ann_metas_all_np = np.append(ann_metas_all_np, ann_metas_all[i])
            ann_imgs_all_np = np.append(ann_imgs_all_np, ann_imgs_all[i])

        logging.info("Saving numpy files...")

        if sql_src is SQL_DENSE:
            save_path = DS2_DENSE_SAVE_PATH + "/numpy_save/dense/"
        elif sql_src is SQL_DENSE_TEST:
            save_path = DS2_DENSE_SAVE_PATH + "/numpy_save/dense_test/"
        elif sql_src is SQL_DENSE_TRAIN:
            save_path = DS2_DENSE_SAVE_PATH + "/numpy_save/dense_train/"
        else:
            save_path = ""
            logging.error("sql_src does not match any of the SQL constants.")

        folder_path = os.path.join(save_path)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            logging.info("No numpy_save directory. Created new one:  %s" % str(folder_path))

        save_to_numpy(np.array(list(categories)), save_path, "categories.npy")
        save_to_numpy(ann_ids_all_np, save_path, "ann_ids_all.npy")
        save_to_numpy(ann_metas_all_np, save_path, "ann_metas_all.npy")
        save_to_numpy(ann_cat_ids_all_np, save_path, "ann_cat_ids_all.npy")
        save_to_numpy(ann_imgs_all_np, save_path, "ann_imgs_all.npy")

        logging.info("Annotations data saved to numpy files.")

        # Write the list of annotation ids that failed to load into a .npy file.

        for f in ann_fails_all:
            ann_fails_all_np = np.append(ann_fails_all_np, f)

        save_to_numpy(ann_fails_all_np, save_path, "ann_fails_all.npy")

    finally:
        logging.info("***  MySQL Connection closed.  ***")
        connection.close()
        cv.destroyAllWindows()


def cnn1(categories):
    logging.info("***  Convolution Neural Network 1 ***")

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

    logging.info("test_x.shape: %s" % test_x.shape)
    logging.info("test_y.shape: %s" % test_y.shape)

    # Train the model using Stochastic Gradient Descent.
    sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=CNN2_EPOCHS)

    # Assess the performance of the neural network.
    pred_y = model.predict(test_x)
    logging.info(classification_report(test_y.argmax(axis=1), pred_y.argmax(axis=1)))

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
    logging.info("***  Convolution Neural Network 2 ***")

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

    logging.info("test_x.shape: %s" % test_x.shape)
    logging.info("test_y.shape: %s" % test_y.shape)

    # Train the model using Stochastic Gradient Descent.
    sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=CNN1_EPOCHS)

    # Assess the performance of the neural network.
    pred_y = model.predict(test_x)
    logging.info(classification_report(test_y.argmax(axis=1), pred_y.argmax(axis=1)))

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


# Loads the annotations for a single image from SQL.
def load_single_image(img_id: int, connection, cursor):
    logging.info("Loading annotations from image %s" % img_id)

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

    ids = []
    cat_ids = []
    metas = []
    imgs = []
    fails = []

    # Load & process the data for every annotation on this image.
    for row in fetch:

        id, cat_id, meta, img = extract_single_annotation(row)

        if img is None:  # The image file may have failed to load.
            logging.error(" !! Img read failed !!   annotation id: %d", id)
            fails.append(id)

        ids.append(id)
        cat_ids.append(cat_id)
        metas.append(meta)
        imgs.append(img)

    return ids, cat_ids, metas, imgs, fails


# Load a single annotation and its metadata from a row of SQL data into the instance numpy datasets.
#   May return 'None' for img if the img file failed to load.
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

    # logging.info("angles: %s \n   o_bbox: %s\n   orientation: %s" % (angles, o_bbox, orientation))

    img_path = DS2_DENSE_SAVE_PATH + "/annotations/" + str(id) + ".jpg"
    if not os.path.exists(img_path):
        logging.error("The annotation image does not exist! Skipped loading this image.   img_path:   %s" % img_path)

    try:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    except cv.Error as e:
        logging.error(e)
        img = None

    return id, cat_id, (height, width, orientation), img


# Load a single numpy array from a binary numpy .npy file.
def load_from_numpy(folder_path: str, file_name: str):
    file_path = os.path.join(folder_path + file_name)

    if not os.path.exists(folder_path):
        logging.error("No numpy save directory! Please save a set of numpy files into folder:   ", folder_path)
        return

    if not os.path.exists(file_path):
        logging.error("No file to load from! Please save the numpy file into:   ", file_path)
        return

    return np.load(file_path)


def log_reg(categories):
    logging.info("***   Logistic Classification   ***")

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

    logging.info(metrics.classification_report(test_y, pred_y, target_names=label_encoder.classes_))

    plt.figure()
    skplt.metrics.plot_confusion_matrix(test_y, pred_y, title=("Logistic Regression Confusion Matrix"), cmap="Reds")
    plt.yticks(label_encoder.fit_transform(label_encoder.classes_), label_encoder.classes_)
    plt.show()

    return test_x, test_y, pred_y


# Save a single numpy array into a binary .npy file.
def save_to_numpy(data: ndarray, folder_path: str, file_name: str):
    folder_path = os.path.join(folder_path)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        logging.log("No numpy_save subdirectory for this database. Created new one:  %s" % folder_path)

    file_path = os.path.join(folder_path + file_name)
    np.save(file_path, data)


# Make the annotations into a standard size of image.
def standardize_annotations():
    pass


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

                info_str = "\nThis data point did not get all the votes" \
                           + ("\n   i: %d" % i) \
                           + ("\n   cnn1_pred_y: %s" % cnn1_pred_y[i]) \
                           + ("\n   cnn2_pred_y: %s" % cnn2_pred_y[i]) \
                           + ("\n   log_reg_pred_y: %s" % log_reg_pred_y[i]) \
                           + ("\n   pred_final: %s" % pred_final[i]) \
                           + ("\n   test_y: %s" % test_y[i])
                logging.info(info_str)
                break

        if vote_tie:
            # If 3-way tie, default to cnn2 results since it has the best stand-alone performance
            pred_final[i] = cnn2_pred_y[i]
            info_str = "\nThis data point had a 3-way tie between models" \
                       + ("\n   i: %d" % i) \
                       + ("\n   cnn1_pred_y: %s" % cnn1_pred_y[i]) \
                       + ("\n   cnn2_pred_y: %s" % cnn2_pred_y[i]) \
                       + ("\n   log_reg_pred_y: %s" % log_reg_pred_y[i]) \
                       + ("\n   pred_final: %s" % pred_final[i]) \
                       + ("\n   test_y: %s" % test_y[i])
            logging.info(info_str)

    return pred_final


# ============== MAIN CODE =========================================================================================

logging.info("=====  SET UP LOGGER  =====")

log_path = os.path.join(ROOT_PATH + '/logs/categorize_annotations.log')
if not os.path.exists(log_path):
    with open(log_path, 'w+') as l:
        pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
logger = logging.getLogger()
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)

logging.info("=====  DATA PREPROCESSING  =====")

# Import the necessary data from sql and save into a set of numpy binary files.
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
# logging.info("\n---- LOG REG PREPROCESSING ----\n")
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
# logging.info("\n---- MODELS ----\n")
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
# logging.info(metrics.classification_report(test_y, pred_final, target_names=label_encoder.classes_))
#
# plt.figure()
# skplt.metrics.plot_confusion_matrix(test_y, pred_final, title="FINAL RESULTS CONFUSION MATRIX", cmap="Reds")
# plt.yticks(label_encoder.fit_transform(label_encoder.classes_), label_encoder.classes_)
# plt.show()

fh.close()

