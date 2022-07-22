"""
@author: Brendan Bassett
@date: 07/20/2022

Assess and load DeepScores V2 dataset from JSON to local MYSQL server.
"""

import json
import mysql.connector
from os import path


def transform(data, save_file_loc: str):

    # The "w" truncates, so it will create new file if one does not exist.
    with open(save_file_loc, 'w') as json_file:

        info_dict = {}
        root_dict = {"info": info_dict}

        # Extract the annotations object, which is a collection of every annotation in this dataset.

        print("\nannotations: \n")

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

        print("\n   number of annotations: ", len(annotations))

        # Extract the annotation_sets object which names each of the sets of annotations.
        print("\nannotation_sets: \n")

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

        print("\ncategories: \n")
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

        print("   number of categories: ", len(categories))

        # Extract the images array which contains data on each image that is annotated.

        print("\nimages: \n")

        images = data['images']
        root_dict['images'] = images

        print("\n   number of images: ", len(images))

        # Extract the info object which describes the dataset.
        print("info: \n")

        info_read = data['info']
        for i in info_read:
            if i == "description":
                info_dict["desc"] = info_read[i]  # This change prevents confusion in the schema
            else:
                info_dict[i] = info_read[i]

            # desc (str), version (str), year (int), contributor (str), date_created (str), url (optional) (str)

        print("Writing to file...")

        root_dict_json = json.dumps(root_dict, indent=4, sort_keys=True)
        json_file.write(root_dict_json)
        json_file.close()

        print("\nSource JSON transformed to new JSON file:" + save_file_loc + "\n")


def populate_local_MySQL(data, connection):
    try:

        cursor = connection.cursor()

        print("-----------------------------------------")
        print("annotations:\n")
        print("    uploading annotations to SQL server...")

        src_annotations = data['annotations']
        out_annotations = []

        for ann in src_annotations:
            annotation = (ann['area'], ann['comments'], ann['id'], ann['img_id'],
                          ann['a_bbox'][0], ann['a_bbox'][1], ann['a_bbox'][2], ann['a_bbox'][3],
                          ann['o_bbox'][0], ann['o_bbox'][1], ann['o_bbox'][2], ann['o_bbox'][3],
                          ann['o_bbox'][4], ann['o_bbox'][5], ann['o_bbox'][6], ann['o_bbox'][7])
            out_annotations.append(annotation)
            # print("category:", category)

        ann_insert_query = "INSERT IGNORE INTO annotations (area, comments, id, img_id," \
                                                            "a_bbox_x0, a_bbox_y0, a_bbox_x1, a_bbox_y1," \
                                                            "o_bbox_x0, o_bbox_y0, o_bbox_x1, o_bbox_y1," \
                                                            "o_bbox_x2, o_bbox_y2, o_bbox_x3, o_bbox_y3) " \
                           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        print("    ann_insert_query:", ann_insert_query)

        cursor.executemany(ann_insert_query, out_annotations)
        connection.commit()

        print("    uploading annotation.cat_id s to SQL server...")

        ann_categories = []
        for ann in src_annotations:
            cat_ids = ann['cat_id']
            for cat_id in cat_ids:
                ann_categories.append((ann['id'], cat_id))

        ann_cat_insert_query = "INSERT IGNORE INTO annotations_categories (ann_id, cat_id) VALUES (%s, %s)"
        print("    ann_cat_insert_query:", ann_cat_insert_query)

        cursor.executemany(ann_cat_insert_query, ann_categories)
        connection.commit()

        print("-----------------------------------------")
        print("categories:\n")
        print("    uploading categories to SQL server...")

        src_categories = data['categories']
        out_categories = []

        for cat in src_categories:
            category = (cat['annotation_set'], cat['color'], cat['id'], cat['name'])
            out_categories.append(category)
            # print("category:", category)


        # cat_insert_query = "INSERT INTO categories (annotation_set, color, id, `name`) " \
        #                    "VALUES (\"%s\", \"%s\", %d, \"%s\")" \
        #                    % (annotation_set, color, id, name)

        cat_insert_query = "INSERT IGNORE INTO categories (annotation_set, color, id, `name`) " \
                           "VALUES (%s, %s, %s, %s)"
        print("    cat_insert_query:", cat_insert_query)

        cursor.executemany(cat_insert_query, out_categories)
        connection.commit()

        print("-----------------------------------------")
        print("images:\n")
        print("    uploading images to SQL server...")

        src_images = data['images']
        out_images = []

        for img in src_images:
            image = (img['filename'], img['height'], img['id'], img['width'])
            out_images.append(image)
            # print("image:", images)

        images_insert_query = "INSERT IGNORE INTO images (id, filename, height, width) " \
                                    "VALUES (%s, %s, %s, %s)"
        print("    images_insert_query:", images_insert_query)

        cursor.executemany(images_insert_query, out_images)
        connection.commit()

        print("    uploading images.ann_id s to SQL server...")

        for img in src_images:
            src_img_ann_ids = img['ann_ids']
            out_img_ann_ids = []

            for ann_id in src_img_ann_ids:
                out_img_ann_ids.append((img['id'], ann_id))

            img_ann_insert_query = "INSERT IGNORE INTO images_ann_ids (image_id, ann_id) VALUES (%s, %s)"
            # print("    out_img_ann_ids:", out_img_ann_ids)
            cursor.executemany(img_ann_insert_query, out_img_ann_ids)

            connection.commit()

        print("-----------------------------------------")
        print("info:\n")
        print("    uploading info to SQL server...")

        src_info = data['info']

        info_insert_query = "INSERT IGNORE INTO info (contributor, `desc`, date_created, version, `year`) " \
                                    "VALUES (\"%s\", \"%s\", %s, \"%s\", \"%s\")" \
                                    % (src_info['desc'], src_info['version'], src_info['year'],
                                       src_info['contributor'], src_info['date_created'])
        print("info_insert_query:", info_insert_query)

        cursor.execute(info_insert_query)
        connection.commit()

    finally:
        print("MySQL Connection closed.")
        connection.close()


# # -------------- MAIN CODE -------------------------------------------------------------------------------------------
#
# # Load the JSON data and transform them to new JSON files with better structure for Schema validation.
#
# print("Loading source data...")
#
# DEEPSCORES_DENSE_TEST = path.join("F://OMR_Datasets/DeepScoresV2_dense/deepscores_test.json")
# DEEPSCORES_DENSE_TRAIN = path.join("F://OMR_Datasets/DeepScoresV2_dense/deepscores_train.json")
#
# test_file = open(DEEPSCORES_DENSE_TEST)
# train_file = open(DEEPSCORES_DENSE_TRAIN)
#
# test_data = json.load(test_file)
# train_data = json.load(train_file)
#
# print("---------------------------------------------")
# print("TEST\n")
# transform(test_data, 'ds2_dense_test.json')
#
# print("---------------------------------------------")
# print("TRAIN\n")
# transform(train_data, 'ds2_dense_train.json')
#
# # --------------------------------------------------------------------------------------------------------------------
#
# # Load revised JSON data.
#
# test_file = open('ds2_dense_test.json')
# test_data = json.load(test_file)
#
# train_file = open('ds2_dense_train.json')
# train_data = json.load(train_file)
#
# # Connect to local MySQL server.
#
# print("\n\n=======TEST DB=================================================================")
# print("initializing connection...")
# test_connection = mysql.connector.connect(user='root', password='MusicE74!',
#                                      host='localhost', db='ds2_dense_test')
# populate_local_MySQL(test_data, test_connection)
#
# print("\n\n=======TRAIN DB=================================================================")
# print("initializing connection...")
# train_connection = mysql.connector.connect(user='root', password='MusicE74!',
#                                      host='localhost', db='ds2_dense_train')
# populate_local_MySQL(train_data, train_connection)
