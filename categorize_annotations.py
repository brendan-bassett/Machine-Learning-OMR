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
import gc
import gzip
import logging
import math
import threading
from datetime import datetime

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
from typing import List

ROOT_PATH = os.path.realpath(os.path.dirname(__file__))
DS2_DENSE_SAVE_PATH = os.path.join("F://OMR_Datasets/DS2_Transformed")

SQL_DENSE = 'ds2_dense'
SQL_DENSE_TEST = 'ds2_dense_test'
SQL_DENSE_TRAIN = 'ds2_dense_train'

NP_SAVE_1_CATS = 'categories_1'
NP_SAVE_1_FAILS = 'fails_1'
NP_SAVE_1_IDS = 'ids_1'
NP_SAVE_1_IMGS = 'imgs%s_1'
NP_SAVE_1_LBL_MATR = 'lbl_matr%s_1'
NP_SAVE_1_METAS = 'metas%s_1'

NP_SAVE_2_CATS = 'categoriess_2'
NP_SAVE_2_IDS_TEST = 'ids_test_2'
NP_SAVE_2_IDS_TRAIN = 'ids_train_2'
NP_SAVE_2_IMGS_TEST = 'imgs_test_2'

ANNOTATION_SET = 'deepscores'  # Annotation category set. Can be 'deepscores' or 'muscima++'

STD_IMG_SHAPE = (25, 25)  # STD_IMG_SHAPE[0] = width      STD_IMG_SHAPE[1] = height
IMG_FLOAT_TYPE = np.float16

CATS_SHAPE = (0, 2)
FAILS_SHAPE = (0)
IDS_SHAPE = (0, 2)
IMGS_SHAPE = (0, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1])
METAS_SHAPE = (0, 3)

BATCH_SIZE = 256        # For both the test and train dataset. There are 684,784 loadable annotations in the train db.
BATCHES_PER_EPOCH = 10  #                                       There are 54,746 loadable annotations in the test db.
VAL_BATCHES = 5
EPOCHS = 1

TRAIN_SIZE = 0.6
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.1

LOG_REG_MAX_ITER = 2000


# ============== CLASSES ===========================================================================================

# Generates batches of data from numpy files.
class DataGeneratorNumpy(keras_utils.Sequence):

    def __init__(self, folder_path: str, num_annotations: int, num_labels: int, shuffle: bool = True):
        self.folder_path = folder_path
        self.num_annotations = num_annotations
        self.num_labels = num_labels
        self.shuffle = shuffle

        self.batch_number = 0
        self.epoch = 0

    def __getitem__(self, index):
        img_batch, label_matr_batch = load_batch_from_numpy(self.folder_path, batch_number=self.batch_number)
        self.batch_number += 1
        return img_batch, label_matr_batch

    def __len__(self):
        return self.num_annotations // BATCH_SIZE


# ============== FUNCTIONS =========================================================================================


# A categorization convolutional neural network. Loads and processes the annotations
# for the test and train databases.
def cnn(ids_test, num_labels: int):

    logging.info("Building neural network...\n")
    # Build the CNN model using sequential dense layers and max pooling.

    model = Sequential()
    model.add(Dense(256, activation='relu'))
    model.add(Conv2D(16, kernel_size=(5, 5), padding="same", activation='relu',
                     input_shape = (BATCH_SIZE, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))

    logging.info("Training model...\n")
    train_gen = DataGeneratorNumpy(get_np_save_path(SQL_DENSE_TRAIN), len(ids_test), num_labels)
    test_gen = DataGeneratorNumpy(get_np_save_path(SQL_DENSE_TEST), len(ids_test), num_labels)
    sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    history = model.fit(train_gen, epochs=EPOCHS, steps_per_epoch=BATCHES_PER_EPOCH,
                        use_multiprocessing=True, validation_data=test_gen, validation_steps=VAL_BATCHES)

    # Assess the performance of the neural network.
    logging.info("Assessing model performance...\n")

    # Print the classification report.
    pred_y = model.predict(test_gen, steps=VAL_BATCHES, use_multiprocessing=True)
    print(classification_report(ids_test[:, 1].argmax(axis=1), pred_y.argmax(axis=1)))

    # Plot the training loss and accuracy.
    plt.figure()
    plt.style.use("ggplot")
    plt.plot(np.arange(0, EPOCHS), history.history["loss"], label="Training Loss")
    plt.plot(np.arange(0, EPOCHS), history.history["val_loss"], label="Testing Loss")
    plt.plot(np.arange(0, EPOCHS), history.history["accuracy"], label="Training Accuracy")
    plt.plot(np.arange(0, EPOCHS), history.history["val_accuracy"], label="Testing Accuracy")
    plt.title("CNN: Training and Testing Loss & Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

    return pred_y.argmax(axis=1)


# Load a single annotation and its metadata from a row of SQL data into the instance numpy datasets.
#   May return 'None' for img if the img file failed to load.
#   Returns (height, width, orientation) for meta, or None if calc_meta = False.
def extract_ann_from_sql_row(row, calc_meta: bool = False):
    id = row[0]
    cat_id = row[1]
    meta = None

    if calc_meta:
        # Calculate META height, width
        height = abs(row[4] - row[2])
        width = abs(row[5] - row[3])

        # Calculate META orientation angle from -pi/4, pi/4.
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

        meta = [height, width, orientation]

    # - end "if calc_meta:"

    # logging.info("angles: %s \n   o_bbox: %s\n   orientation: %s" % (angles, o_bbox, orientation))

    img_path = DS2_DENSE_SAVE_PATH + "/annotations/" + str(id) + ".jpg"
    if not os.path.exists(img_path):
        logging.error("The annotation image does not exist! Skipped loading this image.   img_path:   %s" % img_path)

    try:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    except:
        logging.error(" !! Image failed to load at cv.imread() !!   in extract_ann_from_sql_row()")
        img = None

    return id, cat_id, meta, img


# Get the save path for numpy data.
def get_np_save_path(sql_src: str):
    if sql_src is SQL_DENSE:
        return os.path.join(DS2_DENSE_SAVE_PATH + "/numpy_save/dense_%s/" % ANNOTATION_SET)
    elif sql_src is SQL_DENSE_TEST:
        return os.path.join(DS2_DENSE_SAVE_PATH + "/numpy_save/dense_test_%s/" % ANNOTATION_SET)
    elif sql_src is SQL_DENSE_TRAIN:
        return os.path.join(DS2_DENSE_SAVE_PATH + "/numpy_save/dense_train_%s/" % ANNOTATION_SET)
    else:
        logging.error("sql_src does not match any of the SQL constants.")
        return ""


# Load a batch of annotations from sql and annotation images to numpy. Apply image preprocessing during load process.
# Does not calculate metas, because it must return (X, Y) for CNN.
# RETURNS:  ann_imgs_srt        Numpy ndarray of preprocessed images for the CNN. Sorted in the order of the given ids.
#           train_y_matr        Preprocessed labels as binary matrices.
def load_batch_from_numpy(folder_path, batch_number: int):

    logging.info("Loading batch %s data from numpy file..." % batch_number)

    batch_number_str = f'{batch_number:04d}'
    img_srt_np = load_from_numpy(folder_path, NP_SAVE_1_IMGS % batch_number_str)
    label_matr_np = load_from_numpy(folder_path, NP_SAVE_1_LBL_MATR % batch_number_str)

    return img_srt_np, label_matr_np


# Load a batch of annotations from sql and annotation images to numpy. Apply image preprocessing during load process.
# Does not calculate metas, because it must return (X, Y) for CNN.
# RETURNS:  ann_imgs_srt        Numpy ndarray of preprocessed images for the CNN. Sorted in the order of the given ids.
#           train_y_matr        Preprocessed labels as binary matrices.
def load_batch_from_sql(ids_batch: ndarray, connection, num_categories: int):
    cursor = connection.cursor()

    logging.info("Loading batch data from SQL...")

    ids_str = str(ids_batch[:, 0].tolist()).replace('[', '(').replace(']', ')').replace('\'', '')
    message = "SELECT a.id, ac.cat_id " \
              "FROM annotations a INNER JOIN annotations_categories ac INNER JOIN categories c " \
              "ON (a.id=ac.ann_id AND ac.cat_id=c.id AND c.annotation_set=\'deepscores\')" \
              "WHERE a.id IN %s" % ids_str

    cursor.execute(message)
    connection.commit()
    fetch = cursor.fetchall()

    ids_l = []
    imgs_l = []

    logging.info("Extracting batch data and loading annotation image from SQL row...")

    # Load & process the data for every annotation on this image.
    for row in fetch:

        id, cat_id, meta, img = extract_ann_from_sql_row(row)

        if img is None:  # The image file may have failed to load.
            logging.error("      !! Img read failed !!   annotation batch failed to load.   id: %d" % id)
            return None, None, None, None

        img_std = standardize_ann_img(img)
        if img_std is None:
            logging.error("      !! Img failed to standardize !!   annotation batch failed to load.   id: %d" % id)
            return None, None, None, None
        else:
            img_std = img_std.tolist()

        ids_l.append(id)
        imgs_l.append(img_std)

    logging.info("Sorting the results and finishing preprocessing...")

    # Sort all the loaded data in the order the annotations were given. This ensures we can match
    # up the x and y values after running through the model.

    imgs_l_srt = []
    for ids in ids_batch:
        for j, ann_id in enumerate(ids_l):
            if ids[0] == str(ann_id):
                imgs_l_srt.append(imgs_l[j])
                break

    # Convert y values from integers to binary matrices.
    label_matr_np = keras_utils.to_categorical(ids_batch[:, 1], num_classes=num_categories)

    # Flatten the image data
    img_srt_np = np.array(imgs_l_srt)
    img_srt_np = img_srt_np.reshape((len(img_srt_np), STD_IMG_SHAPE[0], STD_IMG_SHAPE[1], 1)) / 255.0

    return img_srt_np, label_matr_np


# Get the annotation data most useful for our model. Only get the annotations for a single image at a time.
#       Include the annotation's category which corresponds to the annotation set being used in the model.
# RETURNS:  ids_l        2D List of [ann_id, cat_id] rows.
#           metas_l      2D List of [height, width, orientation] where orientation range: (-pi/4, pi/4)
#                           * may return empty list [] if calc_metas == False
#           imgs_l       3D List of standardized images.
#           fails_l      1D List of annotation ids for those that would not load.
def load_by_img(img_id: int, connection, cursor, calc_metas = False):
    logging.info("Loading all annotations from image %s..." % img_id)

    if calc_metas:
        message = "SELECT ann.i, ann_cat.cat_id, " \
                  "a_bbox_x0, a_bbox_y0, a_bbox_x1, a_bbox_y1, " \
                  "o_bbox_x0, o_bbox_y0, o_bbox_x1, o_bbox_y1, " \
                  "o_bbox_x2, o_bbox_y2, o_bbox_x3, o_bbox_y3 FROM annotations ann " \
                  "INNER JOIN annotations_categories ann_cat INNER JOIN categories cat ON (ann.img_id=\'%s\' " \
                  "AND ann.i=ann_cat.ann_id AND ann_cat.cat_id=cat.i AND cat.annotation_set=\'%s\')" \
                  % (img_id, ANNOTATION_SET)
    else:   # calc_metas == False
        message = "SELECT ann.id, ann_cat.cat_id FROM annotations ann " \
                  "INNER JOIN annotations_categories ann_cat INNER JOIN categories cat ON (ann.img_id=\'%s\' " \
                  "AND ann.id=ann_cat.ann_id AND ann_cat.cat_id=cat.id AND cat.annotation_set=\'%s\')" \
                  % (img_id, ANNOTATION_SET)
    cursor.execute(message)
    connection.commit()
    fetch = cursor.fetchall()

    ids_l = []        # 2D list with each row [ann_id, cat_id]
    metas_l = []
    ann_imgs_l = []
    ann_fails_l = []

    # Load & process the data for every annotation on this image.
    for row in fetch:

        i, cat_id, meta, img = extract_ann_from_sql_row(row)

        if img is None:  # The image file may have failed to load.
            logging.error("      !! Img read failed !!   annotation failed to load.   i: %d" % i)
            ann_fails_l.append(i)
            continue

        img_std = standardize_ann_img(img)
        if img_std is None:
            logging.error("      !! Img failed to standardize !!   annotation failed to load.   i: %d" % i)
            ann_fails_l.append(i)
            continue
        else:
            img_std = img_std.astype(IMG_FLOAT_TYPE).tolist()

        ids_l.append([i, cat_id])
        metas_l.append(meta)
        ann_imgs_l.append(img_std)

    return ids_l, metas_l, ann_imgs_l, ann_fails_l


# Load a single numpy array from a binary numpy .npy.gz file.
# RETURNS:  array_np    A single file as a single numpy array.
def load_from_numpy(folder_path: str, name: str, data_shape: tuple = (0), with_indexing=False):
    if not with_indexing:
        file_name = name + '.npy.gz'
        logging.info("Loading numpy file...    %s" % file_name)
        file_path = os.path.join(folder_path + file_name)

        if not os.path.exists(file_path):
            logging.error("No numpy save file! Please save a numpy file:   %s" % file_path)
            return

        file = gzip.GzipFile(file_path, "r")
        load_array = np.load(file, allow_pickle=True)  # allow_pickle=True enables pickled objects, which
        # can be a security risk. We're using it anyways.

        return load_array

    else:   # with_indexing == True
        load_counter = 0
        array_np = np.zeros(data_shape)
        file_name = name + '0000.npy.gz'
        logging.info("Loading numpy file with indexing...    ")
        file_path = os.path.join(folder_path + file_name)

        if not os.path.exists(file_path):
            logging.error("No numpy save file! Please save a set of numpy files into folder:   %s" % folder_path)
            return

        while os.path.exists(file_path):
            logging.info("    loading indexed file...    %s" % file_name)

            file = gzip.GzipFile(file_path, "r")
            load_array = np.load(file, allow_pickle=True)  # allow_pickle=True enables pickled objects, which
            array_np = np.concatenate((array_np, load_array))  # can be a security risk. We're using it anyways.

            load_counter += 1
            index_str = '{0:04}'.format(load_counter)
            file_name = name + index_str + '.npy.gz'
            file_path = os.path.join(folder_path + file_name)

        return array_np


# Reads the image file that corresponds to the given file id.
#       Return: may be null if no image is found.
def load_single_annotation(ann_id: int):
    ann_path = os.path.join(DS2_DENSE_SAVE_PATH + '/annotations/%s' % ann_id + '.jpg')
    if os.path.exists(ann_path):
        return cv.imread(ann_path, cv.IMREAD_GRAYSCALE).astype(dtype=np.uint8)
    else:
        logging.error("    The annotation image does not exist!     ann_path:   %s" % ann_path)
        return


# Loads all of the annotation data from the test and train databases and saves in batches as defined by
# the constant BATCH SIZE. This enables much faster design of the neural network.
def preprocess_data(test_connection, test_cursor, train_connection, train_cursor):
    logging.info("=====  DATA PREPROCESSING - BY BATCH FROM ANNOTATION .JPG FILES TO NUMPY  =====")

    logging.info(" * Loading the categories and transforming to label vectors...")

    # Load the list of every CATEGORY for this annotation set.
    message = "SELECT id, `name` FROM categories WHERE annotation_set=\'deepscores\' ORDER BY id + 0"
    test_cursor.execute(message)
    test_connection.commit()
    fetch = test_cursor.fetchall()

    categories = []
    for row in fetch:
        categories.append([row[0], row[1]])
    categories = np.array(categories)

    # Encode the category ids as integers (vectors).
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(categories[:, 0])

    # Load the list of ids for every IMAGE that is loadable from the TEST dataset.
    logging.info(" * Loading ids for every IMAGE in TEST database...")

    message = "SELECT id FROM images"
    test_cursor.execute(message)
    test_connection.commit()
    fetch = test_cursor.fetchall()

    # Load the list of ids and category ids for every ANNOTATION that is loadable from the TEST dataset.
    logging.info(" * Loading ids for every ANNOTATION in TEST database...")

    message = "SELECT a.id, c.id FROM annotations a INNER JOIN annotations_categories ac INNER JOIN categories c " \
              "ON (a.id = ac.ann_id AND ac.cat_id = c.id AND c.annotation_set = \'%s\' )" \
              "WHERE a.id NOT IN (SELECT ann_id FROM load_fails)" % ANNOTATION_SET
    test_cursor.execute(message)
    test_connection.commit()
    fetch = test_cursor.fetchall()

    ids_test = []
    for row in fetch:
        ids_test.append([row[0], row[1]])
    ids_test = np.array(ids_test)

    # Load the list of ids and category ids for every ANNOTATION that is loadable from the TRAIN dataset.
    logging.info(" * Loading ids for every ANNOTATION in TRAIN database...")

    train_cursor.execute(message)
    train_connection.commit()
    fetch = train_cursor.fetchall()

    ids_train = []
    for row in fetch:
        ids_train.append([row[0], row[1]])
    ids_train = np.array(ids_train)

    # Convert from cat_id to an integer value sequence on range (0, len(labels) - 1)
    ids_enc_test = label_encoder.transform(ids_test[:, 1])
    ids_test[:, 1] = ids_enc_test
    ids_enc_train = label_encoder.transform(ids_train[:, 1])
    ids_train[:, 1] = ids_enc_train

    logging.info("=====  SAVE SQL TO NUMPY  =====")
    # Save databases as numpy files to speed up loading of test data in future development.

    # Determine the path to the folder for saving numpy files.
    path = get_np_save_path(SQL_DENSE_TEST)
    test_path = os.path.join(path)
    path = get_np_save_path(SQL_DENSE_TEST)
    train_path = os.path.join(path)

    logging.info(" * Save TEST data to numpy files at:   %s" % test_path)
    save_sql_db_to_numpy(SQL_DENSE_TEST, test_connection, test_cursor, label_encoder)

    logging.info(" * Save TRAIN data to numpy files at:   %s" % train_path)
    save_sql_db_to_numpy(SQL_DENSE_TRAIN, train_connection, train_cursor, label_encoder)


# Save a single numpy array into a compressed binary .npy.gz file.
def save_to_numpy(data: ndarray, folder_path: str, file_name: str):
    folder_path = os.path.join(folder_path)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        logging.info("No numpy_save subdirectory for this database. Created new one:  %s" % folder_path)

    file_path = os.path.join(folder_path + file_name + ".npy.gz")
    file = gzip.GzipFile(file_path, "w")
    np.save(file=file, arr=data)
    file.close()


# DEPRECATED. This strategy prevented batching, causing RAM overload. This needs to be rearranged into batches matching
#               BATCH_SIZE, but preprocessed appropriately.
# Convert all the annotation images for a database into numpy data arrays and labels. Then save them into flat
# files for later use.
def save_sql_db_to_numpy(sql_src: str, connection, cursor, label_encoder, calc_metas=False):

    logging.info("=====  SAVING SQL DATABASE  %s  =====" % sql_src)

    # Determine the path to the folder for saving numpy files.
    save_path = get_np_save_path(sql_src)
    folder_path = os.path.join(save_path)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        logging.info("No numpy_save directory. Created new one:  %s" % str(folder_path))

    # Load the list of possible categories for the chosen annotation set.
    logging.info("Loading categories...")

    message = "SELECT id, `name` FROM categories WHERE annotation_set = \'%s\'" % ANNOTATION_SET
    cursor.execute(message)
    connection.commit()
    fetch = cursor.fetchall()

    categories = []
    for row in fetch:
        categories.append(row)

    logging.info("Saving categories...")
    save_to_numpy(np.array(list(categories)), save_path, NP_SAVE_1_CATS)

    logging.info("Loading all annotation ids with corresponding category id...")
    # Load the list of annotation ids in this database.
    message = "SELECT a.id, c.id FROM annotations a INNER JOIN annotations_categories ac INNER JOIN categories c " \
              "ON (a.id = ac.ann_id AND ac.cat_id = c.id AND c.annotation_set = \'%s\') " \
              "WHERE a.id NOT IN (SELECT lf.ann_id FROM load_fails lf)" % ANNOTATION_SET
    cursor.execute(message)
    connection.commit()
    fetch = cursor.fetchall()

    ids_l = []
    for row in fetch:
        ids_l.append([row[0], row[1]])
    ids_rdm = np.array(ids_l)

    # Randomize the order of the data to distribute the data more evenly over each batch.
    np.random.shuffle(ids_rdm)

    logging.info("Saving annotation, category id pairs...")
    save_to_numpy(np.array(list(ids_rdm)), save_path, NP_SAVE_1_IDS)

    # Encode the category ids to integer labels.
    ids_enc_test = label_encoder.transform(ids_rdm[:, 1])
    ids_rdm[:, 1] = ids_enc_test

    # Load the annotation data from the SQL database.
    # Load the pixelwise annotation data from jpg files.
    # Consolidate the annotation data into numpy arrays, then save in batches to prevent RAM overload.

    logging.info("Preprocessing annotation images and their label matrices in batches, then save each batch...")

    batch_index = 0
    batches_loaded = 0
    tot_batches = len(ids_rdm) // BATCH_SIZE
    start_time = datetime.now()
    img_start_time = start_time

    for batch_counter in range(0, tot_batches):
        logging.info("Preprocessing and saving annotation batch (%d of %d)" % (batch_counter + 1, tot_batches))

        ids_batch = []
        for i in range(0, BATCH_SIZE):
            ids_batch.append([ids_rdm[batch_index + i][0], ids_rdm[batch_index + i][1]])
        batch_index += BATCH_SIZE
        ids_batch = np.array(ids_batch)

        img_srt_np, label_matr_np = load_batch_from_sql(ids_batch, connection, len(categories))

        batches_loaded += 1
        time_now = datetime.now()
        load_img_diff = (time_now - img_start_time).total_seconds()
        load_all_diff = (time_now - start_time).total_seconds()
        load_dur = (time_now - start_time).total_seconds()
        img_start_time = time_now  # This must be after load_all_diff, load_all_diff, and load_dur
        load_all_dur = load_all_diff * (tot_batches / batches_loaded)

        logging.info("     ~remaining time: %d min   time to load: %d s"
                     % ((load_all_dur - load_dur) / 60, load_img_diff))

        batch_counter_str = f'{batch_counter:04d}'
        # save_to_numpy(metas_np, save_path,  NP_SAVE_1_METAS % batch_counter)
        save_to_numpy(img_srt_np, save_path, NP_SAVE_1_IMGS % batch_counter_str)
        save_to_numpy(label_matr_np, save_path, NP_SAVE_1_LBL_MATR % batch_counter_str)


# Make a single annotation into a standard size of image. Images too large will be resized down.
# Images too small will be padded with black pixels. This should help retain the size of the annotation as much
# as possible, while capping at a reasonably small size for the CNN.
def standardize_ann_img(img: ndarray):
    width = len(img)
    height = len(img[0])

    dx = 0
    dy = 0

    # If the image is smaller than the standard shape, fill in horizontally and/or vertically with white space.
    if width < STD_IMG_SHAPE[0]:
        dx = STD_IMG_SHAPE[0] - width
        width = STD_IMG_SHAPE[0]
    if height < STD_IMG_SHAPE[1]:
        dy = STD_IMG_SHAPE[1] - height
        height = STD_IMG_SHAPE[1]

    # logger.info("fill_img    width: %s   height: %s" % (width, height))
    fill_img = np.zeros(STD_IMG_SHAPE)
    cv.copyMakeBorder(img, (dy // 2), (dy // 2 + dy % 2), (dx // 2), (dx // 2 + dx % 2),
                      cv.BORDER_CONSTANT, dst=fill_img, value=0.)

    if width == STD_IMG_SHAPE[0] and height == STD_IMG_SHAPE[1]:
        if len(fill_img) != STD_IMG_SHAPE[0] or len(fill_img[0]) != STD_IMG_SHAPE[1]:
            logging.error(" !!! FILL ISSUE should be %s!!!    id: %s     shape : (%s, %s)"
                          % (STD_IMG_SHAPE, id, len(fill_img), len(fill_img[0])))
            return None
        else:
            return fill_img  # Store as 8-bit unsigned grayscale arrays to save memory.

    else:
        # If the image is larger than the standard shape, resize the image down.
        # INTER_AREA is the most effective interpolation for image decimation beyond 0.5.
        std_img = cv.resize(img, STD_IMG_SHAPE, fx=0, fy=0, interpolation=cv.INTER_AREA)

        if len(std_img) != STD_IMG_SHAPE[0] or len(std_img[0]) != STD_IMG_SHAPE[1]:
            logging.error(" !!! RESIZE ISSUE should be %s!!!    id: %s     shape : (%s, %s)"
                          % (STD_IMG_SHAPE, id, len(fill_img), len(fill_img[0])))
            return None
        else:
            return std_img


# Uploads the list of annotations that failed to load from .npy.gz file into the SQL database.
# It is unclear why these images fail to load, but it hapens to the same ones every time.
# By saving this list we can prevent issues with loading in batches later.
def upload_fail_list_sql(sql_src: str):
    logging.info("Loading fail list from numpy...")

    fail_ids = load_from_numpy(get_np_save_path(sql_src), 'ann_fails')

    fail_ids_str = ""
    for ann_id in fail_ids:
        ann_id_str = '(' + str(ann_id) + '), '
        fail_ids_str += ann_id_str
    fail_ids_str = fail_ids_str[:-2]  # Remove the last comma and space

    try:
        logging.info("Connecting to MySQL %s database..." % sql_src)
        connection = mysql.connector.connect(user='root', password='HelloPassword0!', host='localhost',
                                             db=sql_src, buffered=True)

        cursor = connection.cursor()

        logging.info("Uploading annotation fail lists to sql...")

        cursor.execute("CREATE TABLE IF NOT EXISTS load_fails (ann_id INT NOT NULL PRIMARY KEY)")
        cursor.execute("TRUNCATE TABLE load_fails")
        cursor.execute("INSERT INTO load_fails VALUES %s" % fail_ids_str)

        connection.commit()

    finally:
        logging.info("MySQL %s database connection closed." % sql_src)
        connection.close()
        cv.destroyAllWindows()

    # ============== MAIN CODE =========================================================================================


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # Prevents "allocation of --- exceeds %10 of free system memory
                                                    # info from tensorflow. Message not an indication of performance."

    logging.info("=====  SET UP LOGGER  =====")

    log_path = os.path.join(ROOT_PATH + '/logs/categorize_annotations.log')
    if not os.path.exists(log_path):  # Create a new empty log file if it doesnt already exist.
        with open(log_path, 'w+') as l:
            pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
    logger = logging.getLogger()
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    logging.info(" * Initializing MySQL DB connection...")

    try:
        test_connection = mysql.connector.connect(user='root', password='HelloPassword0!', host='localhost',
                                                  db=SQL_DENSE_TEST, buffered=True)
        test_cursor = test_connection.cursor()

        train_connection = mysql.connector.connect(user='root', password='HelloPassword0!', host='localhost',
                                                   db=SQL_DENSE_TRAIN, buffered=True)
        train_cursor = train_connection.cursor()

        # logging.info(" * Preprocessing Data...")
        # preprocess_data(test_connection, test_cursor, train_connection, train_cursor)

        # Determine the path to the folder for saved numpy files.
        path = get_np_save_path(SQL_DENSE_TEST)
        test_path = os.path.join(path)
        path = get_np_save_path(SQL_DENSE_TEST)
        train_path = os.path.join(path)

        logging.info(" * Loading numpy files from:   %s" % test_path)

        categories = load_from_numpy(test_path, NP_SAVE_2_CATS, data_shape=CATS_SHAPE)
        ids_test = load_from_numpy(test_path, NP_SAVE_2_IDS_TEST, data_shape=IDS_SHAPE)

        logging.info(" * Loading numpy files from:   %s" % train_path)

        ids_train = load_from_numpy(train_path, NP_SAVE_2_IDS_TRAIN, data_shape=IDS_SHAPE)

        # Encode the category ids as integers (vectors).
        label_encoder = preprocessing.LabelEncoder()
        labels = label_encoder.fit_transform(categories[:, 0])

        logging.info(" * Running CNN categorization model...\n")

        pred_y_cnn = cnn(ids_test, len(labels))
        # test_x, test_y, pred_y_logreg = log_reg(ann_ids_test, ann_ids_train, test_y, train_y)

        # Show the classification report and confusion matrix.
        logging.info("=====   RESULTS AND ANALYSIS   =====")

        # Convert labels to integers.
        logging.info(metrics.classification_report(ids_test[:, 1], pred_y_cnn, target_names=label_encoder.classes_))

        plt.figure()
        skplt.metrics.plot_confusion_matrix(ids_test[:, 1], pred_y_cnn, title="FINAL RESULTS CONFUSION MATRIX", cmap="Reds")
        plt.yticks(label_encoder.fit_transform(label_encoder.classes_), label_encoder.classes_)
        plt.show()

    finally:
        logging.info("Closed MySQL %s database connection." % SQL_DENSE_TEST)
        logging.info("Closed MySQL %s database connection." % SQL_DENSE_TRAIN)
        test_connection.close()
        train_connection.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
