"""
author: Brendan Bassett
date: 08/05/2022

Machine Learning: Optical Music Character Recognition (OMR)
----------------------------------------------------------

Categorizing musical symbols using a convolutional neural network (CNN).

Dataset
-------
    DeepScores V2 (dense version)

Sourced from: https://zenodo.org/record/4012193#.YvGNkHbMLl1

Description:
 The DeepScoresV2 Dataset for Music Object Detection contains digitally rendered images of written sheet music, together
 with the corresponding ground truth to fit various types of machine learning models. A total of 151 Million different
 instances of music symbols, belonging to 135 different classes are annotated. The total Dataset
 contains 255,385 Images. For most researches, the dense version, containing 1714 of the most diverse and interesting
 images, is a good starting point.

 The dataset contains ground in the form of:

    Non-oriented bounding boxes
    Oriented bounding boxes
    Semantic segmentation
    Instance segmentation

The accompaning paper: The DeepScoresV2 Dataset and Benchmark for Music Object Detection
published at ICPR2020 can be found here:

https://digitalcollection.zhaw.ch/handle/11475/20647

(SEE README.MD FOR MORE INFORMATION)
"""

# ============== SETUP =============================================================================================

import cv2 as cv
import gzip
import logging
import math
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import os
import scikitplot as skplt
import tensorflow as tf

from datetime import datetime
from keras.callbacks import Callback
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from numpy.core.records import ndarray
from sklearn import metrics
from sklearn import preprocessing

# ============== CONSTANTS =========================================================================================

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

NUM_BATCHFILES_TRAIN = 2674      # 684,784 loadable annotations in the train db.  / 256  = 2,674.94 Batches per epoch
NUM_BATCHFILES_TEST = 213         # 54,746 loadable annotations in the test db.    / 256  = 213.85 Batches

BATCH_SIZE = 256                        # For both the test and train dataset. They must match.
EPOCH_SIZE = NUM_BATCHFILES_TRAIN       # The number of batches in a single "epoch" of the training dataset
VAL_SIZE = 8                            # The number of batches in the validation phase after each training batch
TEST_SIZE = NUM_BATCHFILES_TEST         # The number of batches in the final test phase after training is complete
EPOCHS = 1


# ============== CLASSES ===========================================================================================

class BatchCallback(Callback):
    """ A callback class to save the accuracy and loss for test and train each batch.

    Source
    ------
    Copied from, then edited:
               https://stackoverflow.com/questions/66394598/how-to-get-history-over-one-epoch-after-every-batch

    Attributes
    ----------
    imgs_test : ndarray
        A preprocessed numpy array of annotation images for validation with each batch. This stays the same each batch.
    lbl_mtr_test : ndarray
        A preprocessed numpy array of label matrices corresponding to each image.
    accuracy : list
        The training accuracy for each batch.
        TODO: Determine whether accuracy is cumulative across all completed batches or just this batch.
    loss : list
        The training loss for each batch.
        TODO: Determine whether loss is cumulative across all completed batches or just this batch.
    val_loss : list
        The test loss for each batch. Generated from the same multi-batch test dataset each time.
    val_acc : list
        The test accuracy for each batch. Generated from the same multi-batch test dataset each time.
   """

    def __init__(self, imgs_test: ndarray, lbl_mtr_test: ndarray):
        """
        Parameters
        ----------
        imgs_test: ndarray
            A multibatch dataset of preprocessed images for validation.
        lbl_mtr_test: ndarray
            A multibatch dataset of preprocessed categories (as label matrices) for validation.
        """

        super(BatchCallback, self).__init__()

        self.imgs_test = imgs_test
        self.lbl_mtr_test = lbl_mtr_test

        self.accuracy = []
        self.loss = []
        self.val_loss = []
        self.val_acc = []

    def on_train_batch_end(self, batch, logs=None):
        """ Saves the performance data for each batch.
        Parameters
        ----------
        batch
            The number of the batch within the epoch (as determined by the model, not this callback).
        logs
            The log of metadata on the completed batch.
        """

        self.accuracy.append(logs.get('accuracy'))
        self.loss.append(logs.get('loss'))

        val_loss_batch, accuracy_batch = self.model.evaluate(self.imgs_test, self.lbl_mtr_test)

        self.val_loss.append(val_loss_batch)
        self.val_acc.append(accuracy_batch)


class DataGeneratorNumpy(tf.keras.utils.Sequence):
    """ A generator class to track and load preprocessed batch data from a single database (e.g. either test or train)

    Attributes
    ----------
    folder_path : str
        The location of the folder containing the preprocessed numpy files.
    num_labels : int
        The number of different labels, or categories possible in this annotation set.
    num_batchfiles: int
        The maximum batch number for this database. Allows randomization across the entire database,
        even for partial epochs. Must be NUM_BATCHFILES_TEST or NUM_BATCHFILES_TRAIN
    shuffle : bool = True
        Whether or not to shuffle the batch numbers before each epoch.
    batch_numbers: ndarray
        The randomized array of batch numbers. Newly randomized each epoch.
    batch_counter : int
        The current batch. Starts at 0.
    epoch_counter : int
        The current epoch. Starts at 1.
   """

    def __init__(self, folder_path: str,
                 num_labels: int,
                 num_batchfiles: int,
                 shuffle: bool = True):
        """
        Parameters:
        ----------
        folder_path: str
            The location of the folder containing the preprocessed numpy files.
        num_labels: int
            The number of different labels, or categories possible in this annotation set.
        max_batch_idx: int
            The maximum batch number for this database. Allows randomization across the entire
            database, even for partial epochs. Must be NUM_BATCHFILES_TEST or NUM_BATCHFILES_TRAIN
        shuffle: bool
            Whether or not to shuffle the batch numbers before each epoch.
        """

        self.folder_path = folder_path
        self.num_labels = num_labels
        self.num_batchfiles = num_batchfiles
        self.shuffle = shuffle

        self.batch_numbers = np.arange(0, num_batchfiles)

        self.batch_counter = 0
        self.epoch_counter = 0

    def __getitem__(self, index):
        """ Load a pair of image arrays and a label matrix arrays for a random batch of data.

        Parameters
        ----------
        index
            Ignored. The generator class itself will iterate over random batches.

        Returns
        -------
        ndarray
            A single batch of preprocessed images.
        ndarray
            A single batch of preprocessed categories as label matrices.
        """

        load_batch_num = self.batch_numbers[self.batch_counter]
        img_batch, label_matr_batch = load_batch_from_numpy(self.folder_path, batch_number=load_batch_num)
        self.batch_counter += 1

        if self.batch_counter >= self.num_batchfiles:
            self.on_epoch_end()

        return img_batch, label_matr_batch

    def __len__(self):
        """ The number of batches in this generator.

        Returns
        -------
        int
            The number of batches. May be less than max_batch_idx for partial epochs.
        """

        return self.num_batchfiles

    def on_epoch_end(self):
        """ Shuffles the batch numbers at the creation of the generator and after each epoch.
        """

        logging.info("   Epoch %d complete." % self.epoch_counter)
        self.batch_counter = 0
        self.epoch_counter += 1
        np.random.shuffle(self.batch_numbers)


# ============== FUNCTIONS =========================================================================================

def cnn(num_labels: int):
    """ A categorization convolutional neural network. Predicts the category of an annotations.

    Loads the data in batches from preprocessed numpy files into the cnn, then tracks performance for each batch
    as it fits to each. Then it makes a final set of predictions after the final batch of the final epoch.

    Parameters
    ----------
    num_labels: int
        The number of labels (categories) in this database.

    Returns
    ----------
    Callback
        The keras callback object for overall perfomance of the model.
    Callback
        The keras callback object for per batch perfomance of the model.
    lbl_mtr_val
        The multibatch test label (category) data. As preprocessed label matrices.
    label_pred
        The multibatch predicted label (category) data for the test batch. As label matrices.
    """

    logging.info("Building neural network...\n")
    # Build the CNN model using sequential dense layers and max pooling.

    model = Sequential()
    model.add(Dense(256, activation='relu'))
    model.add(Conv2D(16, kernel_size=(5, 5), padding="same", activation='relu',
                     input_shape=(BATCH_SIZE, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))

    logging.info("Training model...\n")
    train_gen = DataGeneratorNumpy(get_np_save_path(SQL_DENSE_TRAIN), num_labels, NUM_BATCHFILES_TRAIN)
    test_gen = DataGeneratorNumpy(get_np_save_path(SQL_DENSE_TEST), num_labels, NUM_BATCHFILES_TEST)

    # Load the set of data for validation at the end of each batch.
    imgs_val = np.zeros((0, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1], 1))
    lbl_mtr_val = np.zeros((0, num_labels))
    for batch_counter in range(0, VAL_SIZE):
        imgs_batch, lbl_mtr_batch = test_gen.__getitem__(batch_counter)
        imgs_val = np.concatenate((imgs_val, imgs_batch))
        lbl_mtr_val = np.concatenate((lbl_mtr_val, lbl_mtr_batch))

    test_gen = DataGeneratorNumpy(get_np_save_path(SQL_DENSE_TEST), num_labels, NUM_BATCHFILES_TEST)

    batch_history = BatchCallback(imgs_val, lbl_mtr_val)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    history = model.fit(train_gen, callbacks=[batch_history], epochs=EPOCHS, steps_per_epoch=EPOCH_SIZE,
                        use_multiprocessing=True, validation_data=test_gen, validation_steps=TEST_SIZE)

    # Use the model to make some predictions.
    logging.info("Using model to make test predictions...\n")

    label_pred = model.predict(imgs_val)

    return history, batch_history, lbl_mtr_val, label_pred


def extract_from_row(row: dict, calc_meta: bool = False):
    """ Loads a single annotation and its metadata from a single row of SQL data into the instance numpy datasets.

    Parameters
    ----------
    row: dict
        A single entry (row) of data.
    calc_meta: bool
        Whether or not to calculate metadata (height, width, orientation). Adds significant time to preprocessing.

    Returns
    -------
    int
        annotation id
    str
        category id
    dict
        metadata. If calc_meta is True, metadata = (height, width, orientation).
        Returns None if calc_meta is False.
    img
        Annotation image from .jpg files in /annotations/ folder. May return None if image failed to load.
    """

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

        # Tuples here tend to create issues with conversion from lists to numpy array, so we use list.
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


def get_np_save_path(sql_src: str):
    """ Get the save path for numpy data.

    Parameters
    ----------
    sql_src: str
        The name of the sql database to use as a data source.

    Returns
    -------
    str
        The path for the folder where numpy .npy.gz save files go.
    """

    if sql_src is SQL_DENSE:
        return os.path.join(DS2_DENSE_SAVE_PATH + "/numpy_save/dense_%s/" % ANNOTATION_SET)
    elif sql_src is SQL_DENSE_TEST:
        return os.path.join(DS2_DENSE_SAVE_PATH + "/numpy_save/dense_test_%s/" % ANNOTATION_SET)
    elif sql_src is SQL_DENSE_TRAIN:
        return os.path.join(DS2_DENSE_SAVE_PATH + "/numpy_save/dense_train_%s/" % ANNOTATION_SET)
    else:
        logging.error("sql_src does not match any of the SQL constants.")
        return ""


def load_batch_from_numpy(folder_path: str, batch_number: int):
    """ Load a batch of annotations from sql and annotation images to numpy.
        Does not load metas because it must return (X, Y), not (X, Y, Z) for CNN.

    Parameters
    ----------
    folder_path: str
        The path for the folder where numpy .npy.gz save files go.
    batch_number: int
        The number of the file batch to load. Starts at 0.

    Returns
    -------
    ndarray
        An array of preprocessed images ready for use in the model.
    ndarray
        An array of preprocessed categories in the form of label matrices.
    """

    batch_number_str = f'{batch_number:04d}'
    imgs = load_from_numpy(folder_path, NP_SAVE_1_IMGS % batch_number_str)
    label_matr = load_from_numpy(folder_path, NP_SAVE_1_LBL_MATR % batch_number_str)

    return imgs, label_matr


def load_batch_from_sql(ids_batch: ndarray, connection, num_categories: int):
    """ Load a batch of annotations from sql and annotation images to numpy arrays.
        Preprocessing is applied during this step.

    Parameters
    ----------
    ids_batch
        The id for each annotation in the batch. Returned arrays are each ordered the same as this array of ids.
    connection
        The sql connection for the database to query.
    num_categories
        The number of possible categories for annotations in this annotation set.

    Returns
    -------
        ndarray
            Numpy ndarray of preprocessed images for the CNN. Sorted in the order of the given ids.
        ndarray
            Preprocessed labels as binary matrices.
    """

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

        id, cat_id, meta, img = extract_from_row(row)

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
    label_matr_np = tf.keras.utils.to_categorical(ids_batch[:, 1], num_classes=num_categories)

    # Flatten the image data
    img_srt_np = np.array(imgs_l_srt)
    img_srt_np = img_srt_np.reshape((len(img_srt_np), STD_IMG_SHAPE[0], STD_IMG_SHAPE[1], 1)) / 255.0

    return img_srt_np, label_matr_np


def load_by_img(img_id: int, connection, cursor, calc_metas=False):
    """ Get a single batch of annotation data corresponding to a full page of sheet music.

    Parameters
    ----------
    img_id
    connection
    cursor
    calc_metas

    Returns
    -------
    list
        List of ids. [annotation id, category id]
    list
        Tuples of meta data. (height, width, orientation)
    list
        3D list of annotation images.
    list
        Ids for every annotation that failed to load.
    """

    logging.info("Loading all annotations from image %s..." % img_id)

    if calc_metas:
        message = "SELECT ann.i, ann_cat.cat_id, " \
                  "a_bbox_x0, a_bbox_y0, a_bbox_x1, a_bbox_y1, " \
                  "o_bbox_x0, o_bbox_y0, o_bbox_x1, o_bbox_y1, " \
                  "o_bbox_x2, o_bbox_y2, o_bbox_x3, o_bbox_y3 FROM annotations ann " \
                  "INNER JOIN annotations_categories ann_cat INNER JOIN categories cat ON (ann.img_id=\'%s\' " \
                  "AND ann.i=ann_cat.ann_id AND ann_cat.cat_id=cat.i AND cat.annotation_set=\'%s\')" \
                  % (img_id, ANNOTATION_SET)
    else:  # calc_metas == False
        message = "SELECT ann.id, ann_cat.cat_id FROM annotations ann " \
                  "INNER JOIN annotations_categories ann_cat INNER JOIN categories cat ON (ann.img_id=\'%s\' " \
                  "AND ann.id=ann_cat.ann_id AND ann_cat.cat_id=cat.id AND cat.annotation_set=\'%s\')" \
                  % (img_id, ANNOTATION_SET)
    cursor.execute(message)
    connection.commit()
    fetch = cursor.fetchall()

    ids_l = []  # 2D list with each row [ann_id, cat_id]
    metas_l = []
    ann_imgs_l = []
    ann_fails_l = []

    # Load & process the data for every annotation on this image.
    for row in fetch:

        a_id, cat_id, meta, img = extract_from_row(row)

        if img is None:  # The image file may have failed to load.
            logging.error("      !! Img read failed !!   annotation failed to load.   a_id: %d" % a_id)
            ann_fails_l.append(a_id)
            continue

        img_std = standardize_ann_img(img)
        if img_std is None:
            logging.error("      !! Img failed to standardize !!   annotation failed to load.   a_id: %d" % a_id)
            ann_fails_l.append(a_id)
            continue
        else:
            img_std = img_std.astype(IMG_FLOAT_TYPE).tolist()

        # A tuple here tends to create issues with conversion from ids_l list to numpy arrays, so we use list.
        ids_l.append([a_id, cat_id])
        metas_l.append(meta)
        ann_imgs_l.append(img_std)

    return ids_l, metas_l, ann_imgs_l, ann_fails_l


def load_from_numpy(folder_path: str, name: str):
    """ Load a single numpy array from a binary numpy .npy.gz file.

    Parameters
    ----------
    folder_path
        The path for the folder where numpy .npy.gz save files go.
    name
        The name of the numpy file to load. Do not include .npy.gz

    Returns
    -------
    ndarray
        A single file as a single numpy array.
    """

    file_name = name + '.npy.gz'
    # logging.info("Loading numpy file...    %s" % file_name)
    file_path = os.path.join(folder_path + file_name)

    if not os.path.exists(file_path):
        logging.error("No numpy save file! Please save a numpy file:   %s" % file_path)
        return

    file = gzip.GzipFile(file_path, "r")
    load_array = np.load(file, allow_pickle=True)
    # allow_pickle=True enables pickled objects, which
    # can be a security risk. We're using it anyways.

    return load_array


def load_single_annotation(ann_id: int):
    """ Reads the annotation image file that corresponds to the given file id.

    Parameters
    ----------
    ann_id
        The id of the annotation to load.

    Returns
    -------
    ndarray
        The annotation image. May be None if no image is found.
    """

    ann_path = os.path.join(DS2_DENSE_SAVE_PATH + '/annotations/%s' % ann_id + '.jpg')
    if os.path.exists(ann_path):
        return cv.imread(ann_path, cv.IMREAD_GRAYSCALE).astype(dtype=np.uint8)
    else:
        logging.error("    The annotation image does not exist!     ann_path:   %s" % ann_path)
        return


def preprocess_data(test_connection, test_cursor, train_connection, train_cursor):
    """ Loads all of the annotation data from the test and train databases and saves in batches as defined by
        the constant BATCH SIZE. This enables much faster design of the neural network.

    Parameters
    ----------
    test_connection
        The sql connection for the database to pull TEST data from.
    test_cursor
        The sql cursor for transmitting queries and receiving TEST data.
    train_connection
        The sql connection for the database to pull TRAIN data from.
    train_cursor
        The sql cursor for transmitting queries and receiving TRAIN data.
    """

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
    save_sql_to_numpy(SQL_DENSE_TEST, test_connection, test_cursor, label_encoder)

    logging.info(" * Save TRAIN data to numpy files at:   %s" % train_path)
    save_sql_to_numpy(SQL_DENSE_TRAIN, train_connection, train_cursor, label_encoder)


def save_to_numpy(data: ndarray, folder_path: str, file_name: str):
    """ Save a single numpy array into a compressed binary .npy.gz file.

    Parameters
    ----------
    data
        The numpy array to save to file.
    folder_path
        The path for the folder to save numpy .npy.gz files.
    file_name
        The name of the single file. Do not include .npy.gz
    """

    folder_path = os.path.join(folder_path)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        logging.info("No numpy_save subdirectory for this database. Created new one:  %s" % folder_path)

    file_path = os.path.join(folder_path + file_name + ".npy.gz")
    file = gzip.GzipFile(file_path, "w")
    np.save(file=file, arr=data)
    file.close()


def save_sql_to_numpy(sql_src: str, connection, cursor, label_encoder, calc_metas=False):
    """ Convert all the annotation images for a database into numpy data files ready for use with CNN. Annotation
        images and corresponding labels are loaded, preprocessed, and saved, in batches as indicated by BATCH_SIZE.

    Parameters
    ----------
    sql_src
        The name of the sql database to use a data source.
    connection
        The sql connection for the database to pull data from.
    cursor
        The sql cursor for transmitting queries and receiving data.
    label_encoder
        The object for encoding category ids to integer "labels" starting with 0.
    calc_metas
        Whether to calculate the metadata for each annotation.
    """

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


def standardize_ann_img(img: ndarray):
    """ Make a single annotation into a standard size of image. Images too large will be resized down.
        Images too small will be padded with black pixels. This should help retain the size of the annotation as much
        as possible, while capping at a reasonably small size for the CNN.

    Parameters
    ----------
    img: ndarray
        A single annotation image as loaded from the /annotations/ folder.

    Returns
    -------
    ndarray
        The standardized image.
    """

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


def upload_fail_list_sql(sql_src: str):
    """ Uploads the list of annotations that failed to load from .npy.gz file into the SQL database.
        It is unclear why these images fail to load, but it hapens to the same ones every time.
        By saving this list we can prevent issues with loading in batches later.

    Parameters
    ----------
    sql_src
        The name of the sql database to upload the failure list to.
    """

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


# ================== MAIN CODE =========================================================================================

def main():
    """ Loads, preprocesses, saves annotation data after extraction of pixelwise annotation data from
        a single page sheet music "image". Then runs a deep learning model to predict the category for each.
    """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Prevents "allocation of --- exceeds %10 of free system memory
    # info from tensorflow. Message not an indication of performance."

    logging.info(" ** Setting Up Logger...")

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

    logging.info(" ** Initializing MySQL DB Connection...")

    try:
        test_connection = mysql.connector.connect(user='root', password='HelloPassword0!', host='localhost',
                                                  db=SQL_DENSE_TEST, buffered=True)
        test_cursor = test_connection.cursor()

        train_connection = mysql.connector.connect(user='root', password='HelloPassword0!', host='localhost',
                                                   db=SQL_DENSE_TRAIN, buffered=True)
        train_cursor = train_connection.cursor()

        # logging.info(" ** Preprocessing Data...")
        # preprocess_data(test_connection, test_cursor, train_connection, train_cursor)

        # Determine the path to the folder for saved numpy files.
        path = get_np_save_path(SQL_DENSE_TEST)
        test_path = os.path.join(path)
        path = get_np_save_path(SQL_DENSE_TEST)
        train_path = os.path.join(path)

        logging.info(" ** Loading Numpy Files From:   %s" % test_path)

        categories = load_from_numpy(test_path, NP_SAVE_2_CATS)
        ids_test = load_from_numpy(test_path, NP_SAVE_2_IDS_TEST)

        logging.info(" ** Loading Numpy Files From:   %s" % train_path)

        ids_train = load_from_numpy(train_path, NP_SAVE_2_IDS_TRAIN)

        # Encode the category ids as integers (vectors).
        label_encoder = preprocessing.LabelEncoder()
        labels = label_encoder.fit_transform(categories[:, 0])

        logging.info(" * Running CNN Categorization Model...\n")

        history, batch_history, labels_test, label_predict = cnn(len(labels))

        logging.info(" * Producing Results and Analysis...")

        # Plot the training loss and accuracy over each epoch.
        num_batches = len(batch_history.loss)
        plt.figure()
        plt.style.use("ggplot")
        plt.plot(np.arange(0, num_batches), batch_history.loss, label="Training Loss")
        plt.plot(np.arange(0, num_batches), batch_history.val_loss, label="Testing Loss")
        plt.plot(np.arange(0, num_batches), batch_history.accuracy, label="Training Accuracy")
        plt.plot(np.arange(0, num_batches), batch_history.val_acc, label="Testing Accuracy")
        plt.title("CNN: Training and Testing Loss & Accuracy")
        plt.xlabel("Batch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()

        # # Show the confusion matrix.
        # plt.figure()
        # skplt.metrics.plot_confusion_matrix(labels_test.argmax(axis=1), label_predict.argmax(axis=1),
        #                                     title="FINAL RESULTS CONFUSION MATRIX", cmap="Reds")
        # plt.yticks(label_encoder.fit_transform(label_encoder.classes_), categories[:, 1])
        # plt.show()
        #
        # # Show the classification report.
        # print(metrics.classification_report(labels_test.argmax(axis=1),
        #                                     label_predict.argmax(axis=1),
        #                                     labels=categories[:, 0],
        #                                     target_names=categories[:, 1]))

    finally:
        logging.info("Closed MySQL %s database connection." % SQL_DENSE_TEST)
        logging.info("Closed MySQL %s database connection." % SQL_DENSE_TRAIN)
        test_connection.close()
        train_connection.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
