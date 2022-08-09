
-- This file is for miscelaneous commands for use during the development process.

-- Toggle back and forth between databases.

USE ds2_dense_test;
USE ds2_dense_train;
USE ds2_dense;
SHOW DATABASES;

-- A single annotation and it's image id for use in Python development.

SELECT * FROM annotations WHERE id = 180725;  -- img_id = 38	

-- Get the annotation data most useful for our model. Only get the annotations for a single image at a time. Also 
-- add the annotation's category which corresponds to the annotation set being used in the model.

SELECT ann.id, ann_cat.cat_id, a_bbox_x0, a_bbox_y0, a_bbox_x1, a_bbox_y1, 
o_bbox_x0, o_bbox_y0, o_bbox_x1, o_bbox_y1, o_bbox_x2, o_bbox_y2, o_bbox_x3, o_bbox_y3
FROM annotations ann INNER JOIN annotations_categories ann_cat INNER JOIN categories cat 
ON (ann.img_id=38 AND ann.id=ann_cat.ann_id AND ann_cat.cat_id=cat.id AND cat.annotation_set='deepscores');

-- A small portion of the annotations are too wide or too tall for the CNN. What categories are they and can they be 
-- easily split from the rest of the annotations? Let's say the largest image we can use in the cnn is 100x100.

SELECT count(id) FROM annotations;
SELECT count(id) FROM annotations WHERE (ABS(a_bbox_x1 - a_bbox_x0) > 100 OR ABS(a_bbox_y1 - a_bbox_y0) > 100);
SELECT id, ABS(a_bbox_x1 - a_bbox_x0) AS width FROM annotations WHERE ABS(a_bbox_x1 - a_bbox_x0) > 100;
SELECT id, ABS(a_bbox_y1 - a_bbox_y0) AS height FROM annotations WHERE ABS(a_bbox_y1 - a_bbox_y0) > 100;

-- What are the width and height ranges of each annotation? Is there one category that is much larger than the others?

SELECT cat.`name`, MIN(ABS(a_bbox_x1 - a_bbox_x0)) AS w_min, MAX(ABS(a_bbox_x1 - a_bbox_x0)) AS w_max 
FROM annotations ann INNER JOIN annotations_categories ann_cat INNER JOIN categories cat 
ON (ann.id=ann_cat.ann_id AND ann_cat.cat_id=cat.id) GROUP BY cat.`name` ORDER BY w_min DESC, w_max DESC;

SELECT cat.`name`, MIN(ABS(a_bbox_y1 - a_bbox_y0)) AS h_min, MAX(ABS(a_bbox_y1 - a_bbox_y0)) AS h_max 
FROM annotations ann INNER JOIN annotations_categories ann_cat INNER JOIN categories cat 
ON (ann.id=ann_cat.ann_id AND ann_cat.cat_id=cat.id) GROUP BY cat.`name` ORDER BY h_min DESC, h_max DESC;

-- When we resize the annotations down to less than 100, what happens? Are there many annotations small enough for 
-- the reduction to matter? This shows height vs width vs area for each category that could be problematic.
-- For example, if we resize from 100x100 to 25x25, "problematic" is anything with a width or height less than 4 
-- pixels, so that the entire width or height would become one pixel.
 
SELECT cat.`name`, 
ROUND(AVG(ABS(ann.a_bbox_x1 - ann.a_bbox_x0)), 0) AS avg_width, 
ROUND(AVG(ABS(ann.a_bbox_y1 - ann.a_bbox_y0)), 0) AS avg_height, 
ROUND(AVG(ABS((ann.a_bbox_x1 - ann.a_bbox_x0) * (ann.a_bbox_y1 - ann.a_bbox_y0))), 0) AS avg_area
FROM annotations ann INNER JOIN annotations_categories ann_cat INNER JOIN categories cat
ON ((ABS(ann.a_bbox_x1 - ann.a_bbox_x0) < 4 OR ABS(ann.a_bbox_y1 - ann.a_bbox_y0) < 4) 
AND ann.id = ann_cat.ann_id AND cat_id = cat.id) 
GROUP BY cat_id 
ORDER BY AVG(ABS((ann.a_bbox_x1 - ann.a_bbox_x0) * (ann.a_bbox_y1 - ann.a_bbox_y0))) DESC;

-- But exactly how many annotations could be problematic?

SELECT count(id) FROM annotations WHERE ABS((a_bbox_x1 - a_bbox_x0) * (a_bbox_y1 - a_bbox_y0)) < 16; -- area < 16
SELECT count(id) FROM annotations WHERE ABS(a_bbox_y1 - a_bbox_y0) < 4; -- height < 4
SELECT count(id) FROM annotations WHERE ABS(a_bbox_x1 - a_bbox_x0) < 4; -- width < 4

-- There's a lot of annotations that could be problematic. Maybe we should try to retain sizing for annotations 
-- smaller than 25x25 as we did before with 100x100.

SELECT count(id)s FROM annotations WHERE id NOT IN (SELECT ann_id FROM load_fails);