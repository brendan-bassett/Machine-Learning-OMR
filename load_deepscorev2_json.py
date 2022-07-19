"""
@author: Brendan Bassett
@date: 05/09/2022

ASSESS AND LOAD DEEPSCOREV2 DATASETS FROM JSON
"""

import json
from os import path


def transform(data, save_file_loc: str):

    # The "w" truncates, so it will create new file if one does not exist.
    with open(save_file_loc, 'w') as json_file:
        # Extract the info object which describes the dataset.
        print("info: \n")

        info_dict = {}
        root_dict = {"info": info_dict}
        info_read = data['info']
        for i in info_read:
            if i == "description":
                info_dict["desc"] = info_read[i]  # This change prevents confusion in the schema
            else:
                info_dict[i] = info_read[i]

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
                color = str(cat_read['color'])          # Could be an int or tuple. Convert to string for schema.

                category = {
                    'id': cat_id,
                    'name': name,
                    'annotation_set': annotation_set,
                    'color': color
                }

                categories_array.append(category)

        print("   number of categories: ", len(categories))

        # Extract the images array which contains data on each image that is annotated.

        print("\nimages: \n")

        images = data['images']
        root_dict['images'] = images

        print("\n   number of images: ", len(images))

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
                o_bbox = ann_read['o_bbox']
                cat_id = ann_read['cat_id']
                area = ann_read['area']
                img_id = ann_read['img_id']
                comments = ann_read['comments']

                annotation = {
                    'id': ann_id,
                    'a_bbox': a_bbox,  # array of 4 floats
                    'o_bbox': o_bbox,  # array of 8 floats
                    'cat_id': cat_id,
                    'area': area,
                    'img_id': img_id,
                    'comments': comments
                }

                ann_array.append(annotation)

        root_dict_json = json.dumps(root_dict, indent=4, sort_keys=True)
        json_file.write(root_dict_json)
        json_file.close()

        print("\n   number of annotations: ", len(annotations))

# -------------- MAIN CODE -----------------------------------

# Load the JSON data.

DEEPSCORES_DENSE_TEST = path.join("F://OMR_Datasets/DeepScoresV2_dense/deepscores_test.json")
DEEPSCORES_DENSE_TRAIN = path.join("F://OMR_Datasets/DeepScoresV2_dense/deepscores_train.json")

test_file = open(DEEPSCORES_DENSE_TEST)
train_file = open(DEEPSCORES_DENSE_TRAIN)

test_data = json.load(test_file)
train_data = json.load(train_file)

print("---------------------------------------------")
print("TEST\n")
transform(test_data, 'ds2_dense_test.json')

print("---------------------------------------------")
print("TRAIN\n")
transform(train_data, 'ds2_dense_train.json')
