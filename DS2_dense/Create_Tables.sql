USE ds2_dense_test;
USE ds2_dense_train;
USE ds2_dense;
SHOW DATABASES;

DROP TABLE annotations;
DROP TABLE annotations_categories;
DROP TABLE categories;
DROP TABLE images;
DROP TABLE images_annotations;
DROP TABLE info;

DROP TABLE ann_from_active_image;
DROP TABLE ann_from_unused_image;
DROP TABLE images_with_annotations;
DROP TABLE images_without_annotations;

SHOW TABLES;

CREATE TABLE annotations (id INT NOT NULL PRIMARY KEY, 
							area FLOAT, 
							img_id VARCHAR(255), 
							comments VARCHAR(255),
							a_bbox_x0 FLOAT, 
							a_bbox_y0 FLOAT, 
							a_bbox_x1 FLOAT, 
							a_bbox_y1 FLOAT, 
							o_bbox_x0 FLOAT, 
							o_bbox_y0 FLOAT, 
							o_bbox_x1 FLOAT, 
							o_bbox_y1 FLOAT, 
							o_bbox_x2 FLOAT, 
							o_bbox_y2 FLOAT, 
							o_bbox_x3 FLOAT, 
							o_bbox_y3 FLOAT);
CREATE TABLE annotations_categories (ann_id INT NOT NULL, cat_ID VARCHAR(255));
CREATE TABLE categories (id VARCHAR(255) NOT NULL PRIMARY KEY, `name` VARCHAR(255),  annotation_set VARCHAR(255), color VARCHAR(255));
CREATE TABLE images (id INT NOT NULL PRIMARY KEY, file_name VARCHAR(255),  width INT, height INT);
CREATE TABLE images_annotations (img_id INT NOT NULL, ann_id INT);
CREATE TABLE info (`desc` INT NOT NULL, version VARCHAR(255), `year` INT, contributor VARCHAR(255), date_created VARCHAR(255), url VARCHAR(255));
