"""
@author: Brendan Bassett
@date: 07/20/2022

Split individual annotations into smaller segment image files.
"""
import math
import os
import mysql.connector
import numpy as np
import cv2 as cv


# ============== CONSTANTS =========================================================================================

SRC_PATH = os.path.join("F://OMR_Datasets/DeepScoresV2_dense/images/")
SAVE_PATH = os.path.join("F://OMR_Datasets/DS2_Transformed/annotations/")

SQL_DENSE = 'ds2_dense'
SQL_DENSE_TEST = 'ds2_dense_test'
SQL_DENSE_TRAIN = 'ds2_dense_train'

# ============== FUNCTIONS =========================================================================================


# Displays the image at the given scale. Press any key to close the window.
def show_image(img, img_id: int, scale_x: float = 1, scale_y: float = 1):
    winname = "Image ID: " + str(img_id)
    cv.namedWindow(winname)
    cv.moveWindow(winname, 50, 50)

    resize_shape = (round(img.shape[1] * scale_x), round(img.shape[0] * scale_y))
    img_resize = cv.resize(img, resize_shape, interpolation=cv.INTER_CUBIC)
    cv.imshow(winname, img_resize)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Extract the pixelwise annotations from every image this database refers to.
def extract_all_annotations(sql_db: str):
    print("\n\n***  Initializing MySQL %s DB connection...  ***" % sql_db)

    try:
        connection = mysql.connector.connect(user='root', password='MusicE74!', host='localhost',
                                             db=sql_db, buffered=True)
        cursor = connection.cursor()

        message = "SELECT id FROM images"
        cursor.execute(message)
        connection.commit()
        img_ids = cursor.fetchall()[0]
        print("img_ids", img_ids)

        for img_id in img_ids:
            img = get_img_by_id(connection, img_id[0])
            extract_img_ann(connection, cursor, img, img_id[0])
            print("Completed annotation extraction for image: " + str(img_id[0]))

    finally:
        print("\n\n***  MySQL Connection closed.  ***")
        connection.close()
        cv.destroyAllWindows()


# Extract the pixelwise data for every annotation in a single image. Then save each into a new file.
#       In grouping this task by individual images we avoid potential memory issues.
def extract_img_ann(connection, cursor, img, img_id):

    message = "SELECT * FROM annotations WHERE img_id = \'%s\'" % img_id
    cursor.execute(message)
    connection.commit()
    annotations = cursor.fetchall()

    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    for ann in annotations:

        o_bbox = [(round(ann[8]), round(ann[9])),
                  (round(ann[10]), round(ann[11])),
                  (round(ann[12]), round(ann[13])),
                  (round(ann[14]), round(ann[15]))]

        # Determine the left and rightmost sides.
        left = img.shape[1]  # Initialize these values at their opposite side.
        right = 0
        for i in range(8, 15, 2):
            if ann[i] < left:
                left = math.floor(ann[i])  # Round down for the left side.
            if ann[i] > right:
                right = math.ceil(ann[i])  # Round up for the right side.

        # Determine the
        top = img.shape[0]  # Initialize these values at their opposite side.
        bottom = 0
        for i in range(9, 16, 2):
            if ann[i] < top:
                top = math.floor(ann[i])
            if ann[i] > bottom:
                bottom = math.ceil(ann[i])

        out_bbox = [(left, top), (right, bottom)]
        cv.rectangle(img_color, out_bbox[0], out_bbox[1], (0, 0, 255), 3)

        # These arrays need their shapes to be switched because numpy works in rows, then columns.
        img_out = np.invert(img[out_bbox[0][1]:out_bbox[1][1], out_bbox[0][0]:out_bbox[1][0]])
        mask = np.zeros((img_out.shape[0], img_out.shape[1]), dtype=np.int8)

        # Shift all points of the oriented bounding box to locations within the mask.
        mask_bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for i in range(0, 4):
            mask_bbox[i][0] = o_bbox[i][0] - out_bbox[0][0]
            mask_bbox[i][1] = o_bbox[i][1] - out_bbox[0][1]
        contours = np.array(mask_bbox)

        obbox_mask = cv.fillConvexPoly(mask, contours, 255)
        img_out = cv.bitwise_and(img_out, img_out, mask=obbox_mask)

        filename = str(ann[0]) + '.jpg'
        save_path = SAVE_PATH + filename

        # print("annotation:  ",
        #       "\n   id:", ann[0],
        #       "\n   save_path:", save_path)

        cv.imwrite(save_path, img_out)


# Reads the image file that corresponds to the given file id.
#       Return: may be null if no image is found.
def get_img_by_id(connection, img_id: int):
    cursor = connection.cursor()

    message = "SELECT file_name FROM images WHERE id = \'%s\'" % img_id
    cursor.execute(message)
    connection.commit()
    file_name = cursor.fetchall()[0][0]  # [0][0] Otherwise returns a one-entry array with a one-entry tuple.

    img_path = SRC_PATH + file_name
    if not os.path.exists(img_path):
        print("The image does not exist!\n   img_path:   ", img_path)
        return

    return cv.imread(img_path, cv.IMREAD_GRAYSCALE)


# ============== MAIN CODE =========================================================================================


# Extract all annotations from the sql_dense_test server
print("\n------------------------------------------------------------------\n")
extract_all_annotations(SQL_DENSE_TEST)

print("\n------------------------------------------------------------------\n")
extract_all_annotations(SQL_DENSE_TRAIN)

