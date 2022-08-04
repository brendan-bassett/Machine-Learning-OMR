"""
@author: Brendan Bassett
@date: 07/20/2022

Assess and load DeepScores V2 dataset from JSON to a local MYSQL server.
"""

import json
import logging
import os

import mysql.connector
from os import path

# ============== CONSTANTS =========================================================================================

ROOT_PATH = os.path.realpath(os.path.dirname(__file__))

DS2_DENSE_TEST = path.join("F://OMR_Datasets/DeepScoresV2_dense/deepscores_test.json")
DS2_DENSE_TRAIN = path.join("F://OMR_Datasets/DeepScoresV2_dense/deepscores_train.json")

JSON_DENSE_TEST_TRANS = path.join("F://OMR_Datasets/ds2_Transformed/ds2_dense_test.json")
JSON_DENSE_TRAIN_TRANS = path.join("F://OMR_Datasets/ds2_Transformed/ds2_dense_train.json")

SQL_DENSE = 'ds2_dense'
SQL_DENSE_TEST = 'ds2_dense_test'
SQL_DENSE_TRAIN = 'ds2_dense_train'


# ============== FUNCTIONS =========================================================================================

def transform(source_file: str, save_file_loc: str):
    logging.info("Loading source data...")
    file = open(source_file)
    data = json.load(file)

    logging.info("Transforming source data from json to new, schema-verifiable json... ")

    # The "w" truncates, so it will create new file if one does not exist.
    with open(save_file_loc, 'w') as json_file:

        info_dict = {}
        root_dict = {"info": info_dict}

        # Extract the annotations object, which is a collection of every annotation in this dataset.

        logging.info("annotations... ")

        ann_array = []
        root_dict['annotations'] = ann_array
        annotations = data['annotations']

        for a in range(1, len(annotations)):
            ann_id = str(a)

            if ann_id in annotations:
                ann_read = annotations[ann_id]

                a_bbox = ann_read['a_bbox']
                area = ann_read['area']
                cat_id = ann_read['cat_id']
                comments = ann_read['comments']
                img_id = ann_read['img_id']
                o_bbox = ann_read['o_bbox']

                annotation = {
                    'id': ann_id,
                    'a_bbox': a_bbox,  # [x0, y0, x1, y1] (float)
                    'area': area,
                    'cat_id': cat_id,
                    'comments': comments,
                    'img_id': img_id,
                    'o_bbox': o_bbox  # [x0, y0, x1, y1, x2, y2, x3, y3] (float)
                }

                ann_array.append(annotation)

        logging.info("   number of annotations: %d" % len(annotations))

        # Extract the annotation_sets object which names each of the sets of annotations.
        logging.info("annotation_sets... ")

        ann_sets_array = []
        root_dict['annotation_sets'] = ann_sets_array
        annotation_sets = data['annotation_sets']
        for s in annotation_sets:
            ann_sets_array.append(s)

        # Extract the categories object which contains data on each category of annotation.

        # Each category is a name/value pair with the name equal to the id and the value another object with
        # three name/value pairs. Create a new array with each entry a dictionary representing a category. Then we write
        # them into a new json file containing only an array of categories. There each category id will
        # be a name/value pair.

        logging.info("categories... ")
        categories_array = []
        root_dict["categories"] = categories_array
        categories = data['categories']

        for i in range(1, len(categories)):
            cat_id = str(i)

            if cat_id in categories:
                cat_read = categories[cat_id]

                name = cat_read['name']
                annotation_set = cat_read['annotation_set']
                color = str(cat_read['color'])  # Could be an int or tuple. Convert to string for schema.

                category = {
                    'annotation_set': annotation_set,
                    'color': color,
                    'id': cat_id,
                    'name': name
                }

                categories_array.append(category)

        logging.info("   number of categories: %d" % len(categories))

        # Extract the images array which contains data on each image that is annotated.

        logging.info("images... ")

        images = data['images']
        root_dict['images'] = images

        logging.info("   number of images: %d" % len(images))

        # Extract the info object which describes the dataset.
        logging.info("info... ")

        info_read = data['info']
        for i in info_read:
            if i == "description":
                info_dict["desc"] = info_read[i]  # This change prevents confusion in the schema
            else:
                info_dict[i] = info_read[i]

            # desc (str), version (str), year (int), contributor (str), date_created (str), url (optional) (str)

        logging.info("Writing to file...")

        root_dict_json = json.dumps(root_dict, indent=4, sort_keys=True)
        json_file.write(root_dict_json)
        json_file.close()

        logging.info("Source JSON transformed to new JSON file:" + save_file_loc)


def populate_local_mysql(source_file: str, database: str = SQL_DENSE):
    logging.info("loading...    %s" % source_file)
    file = open(source_file)
    data = json.load(file)

    try:
        connection = mysql.connector.connect(user='root', password='MusicE74!', host='localhost', db=database)
        cursor = connection.cursor()

        # Read every annotation from JSON source files and insert the data into the new MySQL database.

        logging.info("annotations:    uploading annotations to SQL server...")

        src_annotations = data['annotations']
        out_annotations = []

        for ann in src_annotations:
            annotation = (ann['area'], ann['comments'], ann['id'], ann['img_id'],
                          ann['a_bbox'][0], ann['a_bbox'][1], ann['a_bbox'][2], ann['a_bbox'][3],
                          ann['o_bbox'][0], ann['o_bbox'][1], ann['o_bbox'][2], ann['o_bbox'][3],
                          ann['o_bbox'][4], ann['o_bbox'][5], ann['o_bbox'][6], ann['o_bbox'][7])
            out_annotations.append(annotation)
            # logging.info("annotation: %s" % annotation)

        ann_insert_query = "INSERT IGNORE INTO annotations (area, comments, id, img_id, " \
                           "a_bbox_x0, a_bbox_y0, a_bbox_x1, a_bbox_y1, " \
                           "o_bbox_x0, o_bbox_y0, o_bbox_x1, o_bbox_y1, " \
                           "o_bbox_x2, o_bbox_y2, o_bbox_x3, o_bbox_y3) " \
                           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        logging.info("    ann_insert_query: %s" % ann_insert_query)

        cursor.executemany(ann_insert_query, out_annotations)
        connection.commit()

        # Every annotation has an array of categories. Fill a table relating each annotation to its categories.
        # Fill new MySQL database with data from JSON source.
        logging.info("    uploading annotation.cat_id s to SQL server...")

        ann_categories = []
        for ann in src_annotations:
            cat_ids = ann['cat_id']
            for cat_id in cat_ids:
                ann_categories.append((ann['id'], cat_id))

        ann_cat_insert_query = "INSERT IGNORE INTO annotations_categories (ann_id, cat_id) VALUES (%s, %s)"
        logging.info("    ann_cat_insert_query: %s" % ann_cat_insert_query)

        cursor.executemany(ann_cat_insert_query, ann_categories)
        connection.commit()

        # Each annotation category has its own descriptions and metadata.
        # Fill new MySQL database with annotation category data from JSON source.
        logging.info("categories:     uploading categories to SQL server...")

        src_categories = data['categories']
        out_categories = []

        for cat in src_categories:
            category = (cat['annotation_set'], cat['color'], cat['id'], cat['name'])
            out_categories.append(category)
            # logging.info("cat: %s" % cat)

        # cat_insert_query = "INSERT INTO categories (annotation_set, color, id, `name`) " \
        #                    "VALUES (\"%s\", \"%s\", %d, \"%s\")" \
        #                    % (annotation_set, color, id, name)

        cat_insert_query = "INSERT IGNORE INTO categories (annotation_set, color, id, `name`) VALUES (%s, %s, %s, %s)"
        logging.info("    cat_insert_query: %s" % cat_insert_query)

        cursor.executemany(cat_insert_query, out_categories)
        connection.commit()

        # Each image has corresponding metadata.
        # Fill new MySQL database with the metadata from JSON source.
        logging.info("images:    uploading images to SQL server...")

        src_images = data['images']
        out_images = []

        for img in src_images:
            image = (img['filename'], img['height'], img['id'], img['width'])
            out_images.append(image)

        # The source json uses filename instead of file_name as indicated on Deep Scores V2's schema.
        # We are changing it here to match their schema.
        images_insert_query = "INSERT IGNORE INTO images (file_name, height, id, width) VALUES (%s, %s, %s, %s)"
        logging.info("    images_insert_query:", images_insert_query)

        cursor.executemany(images_insert_query, out_images)
        connection.commit()

        # Each image has a list of annotations on it. Fill a table relating each image to its annotations.
        # Fill new MySQL database with the data from JSON source.
        logging.info("    uploading images annotations to SQL server...")

        # This for loop avoids a memory exceeded error. Instead of committing every single row for every image as
        # a single package, commit all the corresponding rows for a single image.
        for img in src_images:
            src_img_annotations = img['ann_ids']
            out_img_annotations = []

            for ann_id in src_img_annotations:
                out_img_annotations.append((img['id'], ann_id))

            img_ann_insert_query = "INSERT IGNORE INTO images_annotations (img_id, ann_id) VALUES (%s, %s)"
            logging.info("    out_img_annotations %s" % out_img_annotations)
            cursor.executemany(img_ann_insert_query, out_img_annotations)

            connection.commit()

        # Each database has its own 'info' metadata.
        # Fill new MySQL database with the metadatadata from JSON source.
        logging.info("info:    uploading info to SQL server...")

        src_info = data['info']

        info_insert_query = "INSERT IGNORE INTO info (contributor, `desc`, date_created, version, `year`) " \
                            "VALUES (\"%s\", \"%s\", %s, \"%s\", \"%s\")" \
                            % (src_info['desc'], src_info['version'], src_info['year'],
                               src_info['contributor'], src_info['date_created'])
        logging.info("info_insert_query:", info_insert_query)

        cursor.execute(info_insert_query)
        connection.commit()

    finally:
        logging.info("MySQL Connection closed.")
        connection.close()


# ============== MAIN CODE =========================================================================================

# Set up logger.

log_path = os.path.join(ROOT_PATH + '/logs/validate_with_schema.log')
if not os.path.exists(log_path):
    with open(log_path, 'w+') as lg:
        pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
logger = logging.getLogger()
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)

# Load the JSON data and transform them to new JSON files with better structure for Schema validation.

logging.info("Transform source json TEST data into schema-verifiable json\n")
transform(DS2_DENSE_TEST, JSON_DENSE_TEST_TRANS)

logging.info("Transform source json TRAIN data into schema-verifiable json\n")
transform(DS2_DENSE_TRAIN, JSON_DENSE_TRAIN_TRANS)

# --------------------------------------------------------------------------------------------------------------------

# Load the revised JSON data and populate new local SQL server with it.

logging.info("Upload revised json TEST data into a local MySQL database.\n")
populate_local_mysql(JSON_DENSE_TEST_TRANS, SQL_DENSE_TEST)

logging.info("Upload revised json TRAIN data into a local MySQL database.\n")
populate_local_mysql(JSON_DENSE_TRAIN_TRANS, SQL_DENSE_TRAIN)
