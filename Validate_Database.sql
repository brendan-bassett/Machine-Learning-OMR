USE ds2_dense;		# This database contains the test and train data (in the dense dataset) COMBINED.

SET GLOBAL net_read_timeout = 300;		# Certain operations here can take more than the 30sec maximum for a local MySQL server.
SET @@GLOBAL.max_connections = 300;		# Set the new timeout to 5 minutes.

-- Are there any images both that appear twice (from being in both the test and train data).
-- 		What about annotations?
SELECT id, COUNT(id) cnt FROM images GROUP BY id HAVING cnt > 1;
SELECT id, COUNT(id) cnt FROM annotations GROUP BY id HAVING cnt > 1;

-- It seems there is no overlap between the test & train databases.

-- Why dont these match? Shouldnt every annotation have an image, and vice-versa?
SELECT DISTINCT img_id FROM annotations ORDER BY LENGTH(img_id), img_id; 
SELECT DISTINCT id FROM images ORDER BY LENGTH(id), id; 

-- Fill a table with ids of every image that actually has annotations in the dataset.
CREATE TABLE IF NOT EXISTS images_with_annotations (img_id INT) SELECT DISTINCT img_id FROM annotations;
SELECT COUNT(img_id) FROM images_with_annotations;

-- Fill a table with ids of every image that does NOT have annotations in the dataset.
CREATE TABLE IF NOT EXISTS images_without_annotations (img_id INT);
INSERT INTO images_without_annotations SELECT img.id FROM images img WHERE img.id NOT IN (SELECT img_id FROM images_with_annotations);
SELECT COUNT(img_id) FROM images_without_annotations;

-- How many annotations referred to in images_annotations do not exist in this dataset?
SELECT count(imgann.ann_id) FROM images_annotations imgann WHERE imgann.ann_id NOT IN (SELECT ann.id FROM annotations ann); 

-- Show all the images that actually have corresponding annotations. There are two ways of doing this, with a join and with a WHERE clause.
SELECT DISTINCT img.id FROM images img INNER JOIN annotations ann ON ann.img_id = img.id ORDER BY LENGTH(img_id), img_id;
SELECT id FROM images WHERE id IN (SELECT DISTINCT img_id FROM annotations) ORDER BY LENGTH(id), id;		# This is about about %30 faster.

-- Show how many images have annotations, then how images there are in total.
SELECT COUNT(id) FROM images WHERE id IN (SELECT DISTINCT img_id FROM annotations);
SELECT COUNT(id) FROM images;

-- Show how many annotations have a corresponding image in the database.
SELECT COUNT(img_id) FROM annotations WHERE img_id IN (SELECT img.id FROM images img);
-- Show how many annotations do NOT have a corresponding image in the database.
SELECT COUNT(DISTINCT img_id) FROM annotations WHERE img_id NOT IN (SELECT img.id FROM images img);

-- For every unused image (ones that have no annotations in the dataset which refer to them), record the annotations that the image refers to.
CREATE TABLE IF NOT EXISTS ann_from_unused_image (img_id INT, ann_id INT);
INSERT INTO ann_from_unused_image SELECT img_ann.img_id, img_ann.ann_id FROM images_annotations img_ann 
		INNER JOIN images_without_annotations img_wo_ann ON img_wo_ann.img_id = img_ann.img_id;
        
-- Now we make sure that every annotation referred to by an unused image, is also not in the dataset. This means that the unused images 
-- can be dropped completely without dropping some of its annotations from the dataset as well.
SELECT ann_id FROM ann_from_unused_image WHERE ann_id IN (SELECT id FROM annotations);

-- For every image that is active (ones that have an annotation which refers to it), record the annotations that the image refers to.
CREATE TABLE IF NOT EXISTS ann_from_active_image (img_id INT, ann_id INT);
INSERT INTO ann_from_active_image SELECT img_ann.img_id, img_ann.ann_id FROM images_annotations img_ann 
		INNER JOIN images_with_annotations img_w_ann ON img_w_ann.img_id = img_ann.img_id;
        
-- For any of the images which is active, do they also refer to any annotations that are not included in the dataset?
SELECT count(img_id) FROM ann_from_active_image WHERE ann_id NOT IN (SELECT id FROM annotations);

-- There is only one active image that refers to annotations not in the database. How many active annotations does it refer to?
SELECT count(ann.id) FROM annotations ann INNER JOIN ann_from_active_image a_active_img ON (a_active_img.img_id = 1054 AND a_active_img.ann_id = ann.id);

-- Conclusion: There are unused images in each dataset. All annotations in the dataset have a corresponding image that is also in the dataset.
-- All images referred to by unused images are NOT in the dataset, indicating that the unused images can all be dropped and still maintain 
-- the integrity of the dataset. There is one image, id=1054 in ds2_dense_train, that refers to 12 annotations included and 237 not included 
-- in the dataset. We could drop that image and drop its remaining 12 annotations for simplicity.

-- Delete the unused images, then drop the temporary tables we created.
DELETE FROM images img WHERE img.id IN (SELECT img_id FROM images_without_annotations);
DELETE FROM images img WHERE img.id = 1054;
DELETE FROM annotations ann WHERE ann.img_id = 1054;

DROP TABLE IF EXISTS images_without_annotations;
DROP TABLE IF EXISTS images_with_annotations;
DROP TABLE IF EXISTS ann_from_unused_image;
DROP TABLE IF EXISTS ann_from_active_image;