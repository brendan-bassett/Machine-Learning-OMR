"""
@author: Brendan Bassett
@date: 07/20/2022

Split individual annotations into smaller segment image files.
"""
import mysql.connector



def get_annotations_from_image(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM images")
    all_img_ids = cursor.fetchall()

    # print("all_img_ids:\n", all_img_ids)

    cursor.execute("SELECT DISTINCT img_id FROM annotations ORDER BY LENGTH(img_id), img_id")
    all_ann_img_ids = cursor.fetchall()

    print("all_ann_img_ids:\n", all_ann_img_ids)

    ann_ids_message = "SELECT id FROM annotations WHERE img_id = %s" % "38"
    cursor.execute(ann_ids_message)
    connection.commit()
    img_ann = cursor.fetchall()
    print("ann_ids message:", ann_ids_message, "\n      img_ann:", img_ann)

    for ann in img_ann:
        annotation_message = "SELECT * FROM annotations WHERE id = %s" % ann[0]
        cursor.execute(annotation_message)
        connection.commit()
        annotation = cursor.fetchall()
        print("annotation:", annotation, "\n   annotation_message:", annotation_message)


# ============== MAIN CODE =========================================================================================

print("Loading source data...")

# Connect to local MySQL server.
try:
    print("\n------------------------------------------------------------------\n")
    print("initializing MySQL ds2_dense_test DB connection...")
    test_connection = mysql.connector.connect(user='root', password='MusicE74!', host='localhost',
                                              db='ds2_dense_test', buffered=True)
    get_annotations_from_image(test_connection)

    print("\n------------------------------------------------------------------\n")
    print("initializing MySQL ds2_dense_train DB connection...")
    train_connection = mysql.connector.connect(user='root', password='MusicE74!', host='localhost',
                                              db='ds2_dense_train', buffered=True)
    get_annotations_from_image(train_connection)



finally:
    print("MySQL Connection closed.")
    test_connection.close()
