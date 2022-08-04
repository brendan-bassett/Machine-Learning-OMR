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

ANNOTATION_SET = 'deepscores'  # Annotation category set. Can be 'deepscores' or 'muscima++'

STD_IMG_SHAPE = (100, 100)  # STD_IMG_SHAPE[0] = width      STD_IMG_SHAPE[1] = height

ANN_LOAD_LIMIT = 20000  # The number of annotations to save at a time.
BATCH_LOAD_SIZE = 256

CNN1_EPOCHS = 4
CNN2_EPOCHS = 5

TRAIN_SIZE = 0.6
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.1

LOG_REG_MAX_ITER = 2000


# ============== VARIABLES =========================================================================================


# ============== FUNCTIONS =========================================================================================

# Generate batches of annotations from a randomized list of ids.
def batch_generator(all_ids: ndarray, connection):
    batch_ids = []
    while True:
        np.random.shuffle(all_ids)
        for i in all_ids:
            batch_ids.append(i)
            if len(batch_ids) == BATCH_LOAD_SIZE:
                yield load_batch(connection, batch_ids)
                batch_ids = []


def cnn(labels, batch_generator, ):

    # Build the CNN model using sequential dense layers and max pooling.
    logging.info("Building CNN categorization model...")

    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(24, kernel_size=(5, 5), padding="same", activation='relu', input_shape=STD_IMG_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(labels), activation='softmax'))

    logging.info("test_x.shape: %s" % test_x.shape)
    logging.info("test_y.shape: %s" % test_y.shape)

    # Train the model using Stochastic Gradient Descent.
    logging.info("Training model...")

    sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=CNN2_EPOCHS)

    # Assess the performance of the neural network.
    logging.info("Assessing model performance...")

    # Print the classification report.
    pred_y = model.predict(test_x)
    logging.info(classification_report(test_y.argmax(axis=1), pred_y.argmax(axis=1)))

    # Plot the training loss and accuracy.
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


# Load a single annotation and its metadata from a row of SQL data into the instance numpy datasets.
#   May return 'None' for img if the img file failed to load.
#   Returns (height, width, orientation) for meta.
def extract_ann_from_sql_row(row):
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

    return id, cat_id, [height, width, orientation], img


# Get the save path for numpy data.
def get_np_save_path(sql_src: str):
    if sql_src is SQL_DENSE:
        return DS2_DENSE_SAVE_PATH + "/numpy_save/dense/"
    elif sql_src is SQL_DENSE_TEST:
        return DS2_DENSE_SAVE_PATH + "/numpy_save/dense_test/"
    elif sql_src is SQL_DENSE_TRAIN:
        return DS2_DENSE_SAVE_PATH + "/numpy_save/dense_train/"
    else:
        logging.error("sql_src does not match any of the SQL constants.")
        return ""


# Load a batch of annotations from sql and annotation images to numpy. Apply image preprocessing during load process.
def load_batch(connection, ids_batch: List[int]):
    cursor = connection.cursor()

    logging.info("Loading batch from SQL and image files...")

    message = "SELECT ann.id, ann_cat.cat_id, " \
              "a_bbox_x0, a_bbox_y0, a_bbox_x1, a_bbox_y1, " \
              "o_bbox_x0, o_bbox_y0, o_bbox_x1, o_bbox_y1, " \
              "o_bbox_x2, o_bbox_y2, o_bbox_x3, o_bbox_y3 FROM annotations ann " \
              "INNER JOIN annotations_categories ann_cat INNER JOIN categories cat ON (ann.id IN \'%s\' " \
              "AND ann.id=ann_cat.ann_id AND ann_cat.cat_id=cat.id AND cat.annotation_set=\'%s\')" \
              % (ids_batch, ANNOTATION_SET)
    cursor.execute(message)
    connection.commit()
    fetch = cursor.fetchall()

    ann_ids = []
    ann_cat_ids = []
    ann_metas = []
    ann_imgs = []

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

        ann_ids.append(id)
        ann_cat_ids.append(cat_id)
        ann_metas.append(meta)
        ann_imgs.append(img_std.tolist())

    return np.array(ann_ids), np.array(ann_cat_ids), np.array(ann_metas), np.array(ann_imgs)


# Get the annotation data most useful for our model. Only get the annotations for a single image at a time.
#       Include the annotation's category which corresponds to the annotation set being used in the model.
def load_by_img(img_id: int, connection, cursor):
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

    ann_ids = []
    ann_cat_ids = []
    ann_metas = []
    ann_imgs = []
    ann_fails = []

    # Load & process the data for every annotation on this image.
    for row in fetch:

        id, cat_id, meta, img = extract_ann_from_sql_row(row)

        if img is None:  # The image file may have failed to load.
            logging.error("      !! Img read failed !!   annotation failed to load.   id: %d" % id)
            ann_fails.append(id)
            continue

        img_std = standardize_ann_img(img)
        if img_std is None:
            logging.error("      !! Img failed to standardize !!   annotation failed to load.   id: %d" % id)
            ann_fails.append(id)
            continue

        ann_ids.append(id)
        ann_cat_ids.append(cat_id)
        ann_metas.append(meta)
        ann_imgs.append(img_std.astype(np.uint8).tolist())

    return ann_ids, ann_cat_ids, ann_metas, ann_imgs, ann_fails


# Load a single numpy array from a binary numpy .npy file.
def load_from_numpy(folder_path: str, name: str, data_shape: tuple = (0)):
    if data_shape == (0):
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

    else:
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
            print(load_array)
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


# Save a single numpy array into a compressed binary .npy.gz file.
def save_to_numpy(data: ndarray, folder_path: str, file_name: str):
    folder_path = os.path.join(folder_path)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        logging.log("No numpy_save subdirectory for this database. Created new one:  %s" % folder_path)

    file_path = os.path.join(folder_path + file_name + ".npy.gz")
    file = gzip.GzipFile(file_path, "w")
    np.save(file=file, arr=data)
    file.close()


# Convert all the pixelwise annotations for a database into numpy data arrays and labels. Then save them into flat
# files for later use.
def save_sql_db_to_numpy(sql_src: str):

    # Determine the path to the folder for saving numpy files.
    save_path = get_np_save_path(sql_src)
    folder_path = os.path.join(save_path)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        logging.info("No numpy_save directory. Created new one:  %s" % str(folder_path))

    logging.info("Initializing MySQL %s DB connection..." % sql_src)
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

        categories = []
        for row in fetch:
            categories.append(row)

        logging.info("   saving categories")
        save_to_numpy(np.array(list(categories)), save_path, "categories")

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

        # Load the annotation data from the SQL database.
        # Load the pixelwise annotation data from jpg files.
        # Consolidate the annotation data into numpy arrays, then save in batches to prevent RAM overload.

        ann_ids_by_img = np.zeros(0)
        ann_cat_ids_by_img = np.zeros(0)
        ann_metas_by_img = np.zeros((0, 3))
        ann_imgs_by_img = np.zeros((0, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1]))
        ann_fails_by_img = np.zeros(0)

        save_counter = 0
        imgs_loaded = 0
        tot_imgs = len(img_ids)
        start_time = datetime.now()
        img_start_time = start_time

        logging.info("Loading annotations...")

        for img_id in img_ids:
            logging.info("Loading annotations from image %s" % img_id)

            ann_ids, ann_cat_ids, ann_metas, ann_imgs, ann_fails = load_by_img(img_id, connection, cursor)

            # Even though concatenate here is slow (especially when the arrays get larger), we use numpy
            # to prevent memory errors. Large Python lists of ints and floats use huge amounts of RAM.
            ann_ids_by_img = np.concatenate((ann_ids_by_img, ann_ids))
            ann_cat_ids_by_img = np.concatenate((ann_cat_ids_by_img, ann_cat_ids))
            ann_metas_by_img = np.concatenate((ann_metas_by_img, ann_metas))
            ann_imgs_by_img = np.concatenate((ann_imgs_by_img, ann_imgs))
            ann_fails_by_img = np.concatenate((ann_fails_by_img, ann_fails))

            imgs_loaded += 1
            time_now = datetime.now()
            load_img_diff = (time_now - img_start_time).total_seconds()
            load_all_diff = (time_now - start_time).total_seconds()
            load_dur = (time_now - start_time).total_seconds()
            img_start_time = time_now  # This must be after load_all_diff, load_all_diff, and load_dur
            load_all_dur = load_all_diff * (tot_imgs / imgs_loaded)

            logging.info("   imgs loaded: (%d of %d)   anns loaded: %d   ~remaining time: %d min   time to load: %d s"
                         % (imgs_loaded, tot_imgs, len(ann_metas_by_img),
                            (load_all_dur - load_dur) / 60, load_img_diff))

            # The meta and imgs arrays could possibly become too large, causing a RAM exceeded error.
            # We batch save those two arrays. The rest are saved after all annotations have been loaded.
            # Also save the final batch regardless.

            if len(ann_metas_by_img) > ANN_LOAD_LIMIT or imgs_loaded == tot_imgs:
                index = '{0:04}'.format(save_counter)
                logging.info("Load limit reached. Batch saving numpy files with index: %s" % index)

                save_to_numpy(ann_metas_by_img, save_path, "ann_metas%s" % index)
                save_to_numpy(ann_imgs_by_img, save_path, "ann_imgs%s" % index)

                ann_metas_by_img = np.zeros((0, 3))
                ann_imgs_by_img = np.zeros((0, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1]))
                save_counter += 1

                logging.info("   numpy files saved and arrays reset")

            gc.collect()  # Ask the garbage collector to remove unreferenced data ASAP. Just in case.

        # After all the annotations are loaded and batch-saved, save the remaining data to .npy

        save_to_numpy(ann_ids_by_img, save_path, "ann_ids")
        save_to_numpy(ann_cat_ids_by_img, save_path, "ann_cat_ids")
        save_to_numpy(ann_fails_by_img, save_path, "ann_fails")

        logging.info("Annotations data saved to numpy files.")

    finally:
        logging.info("Closed MySQL %s database connection." % sql_src)
        connection.close()
        cv.destroyAllWindows()


# Make a single annotation into a standard size of image. Images too large will be resized down.
# Images too small will be padded with black pixels. This should help retain the size of the annotation as much
# as possible, while capping at a reasonably small size for the CNN.
def standardize_ann_img(img):
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
    fill_img = np.zeros((STD_IMG_SHAPE[0], STD_IMG_SHAPE[1]))
    cv.copyMakeBorder(img, (dy // 2), (dy // 2 + dy % 2), (dx // 2), (dx // 2 + dx % 2),
                      cv.BORDER_CONSTANT, dst=fill_img, value=0.)

    if width == STD_IMG_SHAPE[0] and height == STD_IMG_SHAPE[1]:
        if len(fill_img) != 100 or len(fill_img[0]) != 100:
            logging.error(" !!! This image is not 100x100 !!! FILL ISSUE   id: %s     shape : (%s, %s)"
                          % (id, len(fill_img), len(fill_img[0])))
            return None
        else:
            return fill_img  # Store as 8-bit unsigned grayscale arrays to save memory.

    else:
        # If the image is larger than the standard shape, resize the image down.
        # INTER_AREA is the most effective interpolation for image decimation beyond 0.5.
        std_img = cv.resize(img, STD_IMG_SHAPE, fx=0, fy=0, interpolation=cv.INTER_AREA)

        if len(std_img) != 100 or len(std_img[0]) != 100:
            logging.error(" !!! This image is not 100x100 !!! RESIZE ISSUE   id: %s     shape : (%s, %s)"
                          % (id, len(std_img), len(std_img[0])))
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
    fail_ids_str = fail_ids_str[:-2]        # Remove the last comma and space

    try:
        logging.info("Connecting to MySQL %s database..." % sql_src)
        connection = mysql.connector.connect(user='root', password='MusicE74!', host='localhost',
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

logging.info("=====  SET UP LOGGER  =====")

log_path = os.path.join(ROOT_PATH + '/logs/categorize_annotations.log')
if not os.path.exists(log_path):        # Create a new empty log file if it doesnt already exist.
    with open(log_path, 'w+') as l:
        pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
logger = logging.getLogger()
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)

# logging.info("=====  DATA PREPROCESSING - BY ALL INTO .NPZ.GZ FILES  =====")
#
# # Import the necessary data from sql and save into a set of numpy binary files.
# # This strategy is not feasibly, as it prevents the randomization of annotations from one epoch to the next without
# # loading every single numpy file. Instead, we will process in batches below.
#
# logging.info(" * Converting SQL and annotation image data to numpy arrays")
# convert_sql_to_numpy(SQL_DENSE_TEST)
# convert_sql_to_numpy(SQL_DENSE_TRAIN)
#
# logging.info(" * Upload the list of annotations that failed to load into SQL.")
# upload_fail_list_sql(SQL_DENSE_TEST)
# upload_fail_list_sql(SQL_DENSE_TRAIN)
#
# # The full size of the training data, when loaded into 64-bit float numpy arrrays is > 18GB in RAM.
# # This strategy simply is not feasible on a single-CPU consumer laptop. Instead we will process
# # annotations in batches below.
# logging.info(" * Load the full data from numpy files.")
# categories = load_from_numpy(get_np_save_path(SQL_DENSE_TEST), "categories")
# test_x = load_from_numpy(get_np_save_path(SQL_DENSE_TEST), "ann_imgs",
#                          data_shape=(0, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1]))
# test_y = load_from_numpy(get_np_save_path(SQL_DENSE_TEST), "ann_cat_ids")
# train_x = load_from_numpy(get_np_save_path(SQL_DENSE_TRAIN), "ann_imgs",
#                           data_shape=(0, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1]))
# train_y = load_from_numpy(get_np_save_path(SQL_DENSE_TRAIN), "ann_cat_ids")

logging.info("=====  DATA PREPROCESSING - BY BATCH FROM ANNOTATION .JPG FILES  =====")

logging.info(" * Loading the categories and transforming to label vectors...")

# Load the list of every category in this annotation set.
categories = load_from_numpy(get_np_save_path(SQL_DENSE_TEST), "categories")

# Encode the labels as integers (vectors).
label_encoder = preprocessing.LabelEncoder()
labels = label_encoder.fit_transform(categories[:, 0])

# Load the list of ids for every annotation that is loadable from the test & train datasets.
logging.info(" * Loading ids for every annotation in each dataset...")

ann_ids_test = load_from_numpy(get_np_save_path(SQL_DENSE_TEST), "ann_ids")
ann_ids_train = load_from_numpy(get_np_save_path(SQL_DENSE_TRAIN), "ann_ids")


# Flatten the input data.
test_x = test_x.reshape((test_x.shape[0], STD_IMG_SHAPE[0], STD_IMG_SHAPE[1])).astype('float') / 255.0
train_x = train_x.reshape((train_x.shape[0], STD_IMG_SHAPE[0], STD_IMG_SHAPE[1])).astype('float') / 255.0


# Convert y values from integers to binary matrices.
test_y = keras_utils.to_categorical(test_y, num_classes=len(categories))
# train_y = keras_utils.to_categorical(train_y, num_classes=len(categories))

logging.info("=====   MODELS   =====")

# Run the models.

pred_y_cnn1 = cnn1(test_x, test_y, train_x, train_y)
test_x, test_y, pred_y_cnn2 = cnn2()
test_x, test_y, pred_y_logreg = log_reg()

pred_final = vote(test_x, test_y, pred_y_cnn1, pred_y_cnn2, pred_y_logreg)

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
