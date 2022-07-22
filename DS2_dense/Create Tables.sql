USE ds2_dense_test;
USE ds2_dense_train;

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
CREATE TABLE categories (id VARCHAR(255) NOT NULL, `name` VARCHAR(255),  annotation_set VARCHAR(255), color VARCHAR(255));
CREATE TABLE images (id INT NOT NULL PRIMARY KEY, filename VARCHAR(255),  width INT, height INT);
CREATE TABLE images_ann_ids (image_id INT NOT NULL, ann_id INT);
CREATE TABLE info (`desc` INT NOT NULL, version VARCHAR(255), `year` INT, contributor VARCHAR(255), date_created VARCHAR(255), url VARCHAR(255));

SELECT * FROM annotations;
SELECT * FROM annotations_categories;
SELECT * FROM categories;
SELECT * FROM images;
SELECT * FROM images_ann_ids; 
SELECT * FROM info; 

TRUNCATE TABLE annotations; 
TRUNCATE TABLE annotations_categories;
TRUNCATE TABLE categories;
TRUNCATE TABLE images;
TRUNCATE TABLE images_ann_ids;
TRUNCATE TABLE info;
