# Machine Learning: Optical Music Character Recognition (OMR)

Categorizing musical symbols using a convolutional neural network (CNN).

Created by Brendan Bassett

Originally built for Machine Learning final project May 10, 2022.
Drastically extended and revised August 8, 2022.

## Libraries Used

MySQL, NumPy, Scikit-learn, Tensorflow, Matplotlib, OpenCV, Logging

## Results

loss: 1.1498 - accuracy: 0.6985 - val_loss: 0.9491 - val_accuracy: 0.7137

<img alt="CNN Train vs Test Loss & Accuracy" width="800" height="auto" src="https://github.com/brendan-bassett/Machine-Learning-OMR/blob/master/plots/TrainTest_LossAccuracy.png"/>

Final results after 256 batches of 256 annotations each in Training batch (1/10 epoch).
Test size 20 batches of 256 annotations each.

## Dataset

### DeepScores V2 (dense version)

Sourced from: https://zenodo.org/record/4012193#.YvGNkHbMLl1

### Description
 The DeepScoresV2 Dataset for Music Object Detection contains digitally rendered images of written sheet music, together
 with the corresponding ground truth to fit various types of machine learning models. A total of 151 Million different
 instances of music symbols, belonging to 135 different classes are annotated. The total Dataset
 contains 255,385 Images. For most researches, the dense version, containing 1714 of the most diverse and interesting
 images, is a good starting point.

### Structure
 The dataset contains ground in the form of:

    Non-oriented bounding boxes
    Oriented bounding boxes
    Semantic segmentation
    Instance segmentation

### Source Paper
The accompaning paper: The DeepScoresV2 Dataset and Benchmark for Music Object Detection published at ICPR2020 can be 
found here:

https://digitalcollection.zhaw.ch/handle/11475/20647

## Motivation

Most of the sheet music that musicians typically use is digitized from a physical copy.
Often it is a poorly taken photo of a book, or a photocopy, or even a copy of a copy. This means
that the vast majority of musicians’ digital collections are of poor quality and difficult to read.
Originals are often in bad condition or not even present, so the solution is rarely to make a
better scan in the first place. Wouldn’t it be then nice to convert all of these poor copies to much
cleaner, more readable versions?

We could use some image processing techniques to remove artifacts and improve
binarization (conversion to black and white), but this only goes so far. Since the dawn of the
digital era, musicians have sought a method to automate the “reading” of music itself. This
would enable us to extract the musical information so that the sheet music can be reproduced in
significantly better visual detail. Think of text and how each letter of the alphabet can be
expressed as tiny packets of data, rather than storing an actual image of each letter. Musical
information can be stored in similar ways, through formats like Music XML and MIDI. However,
given the complexity of musical notation, conversion to one these formats from a physical
document has proven to be very difficult.

What I am describing is called Optical Music Recognition (OMR). Only recently has it
reached a level of accuracy which is meaningful. Some of the issues that OMR still struggles
with is the spacial-relational nature of musical notation. Most notation is written on top of a set of
lines call the staff, which has to be removed before individual characters can be recognized.
Notes of the same pitch are located at the same vertical location on the staff lines (in general,
but not always), so this information must be recognized and stored for analysis. Notes of the
same duration can often be grouped together with bars, or with the flag up or down, or with no
flag at all. Those details can sometimes be meaningful, and sometimes not. Worst of all for the
software developer, much music only exists as handwritten copies, which makes each symbol
appear wildly different from composer to composer.

Neural networks are the essential technology that OMR has needed to be remotely
useful. It enables the software to accommodate for the vast diversity of musical notation. In this
project I take a dataset of common musical characters, already removed from the staffs they are
on, and seek to categorize them.
