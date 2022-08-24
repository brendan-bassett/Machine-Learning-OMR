# Machine Learning: Optical Music Character Recognition (OMR)

Categorizing musical symbols using a convolutional neural network (CNN).

Created by Brendan Bassett

Originally built for Machine Learning final project May 10, 2022.
Drastically extended and revised August 8, 2022.

## Libraries Used

MySQL, NumPy, Scikit-learn, Tensorflow, Matplotlib, OpenCV, Logging

## Results

Training size 2674 batches of 256 annotations each.
Test size 213 batches of 256 annotations each.

Epoch 1/2

2674/2674 9900s 4s/step - loss: 0.6532 - accuracy: 0.7812 - val_loss: 0.6786 - val_accuracy: 0.7825

Epoch 2/2
2674/2674 27223s 10s/step - loss: 0.6580 - accuracy: 0.7715 - val_loss: 0.6600 - val_accuracy: 0.7841


(Precision, Recall, f1-score, and support for each category and in averages listed at the bottom of this README)

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

## More Results

| ***category*** | **precision** | **recall** | **f1-score** | **support** | 
| --: | ---------: | ------: | --------: | -------: |
| *brace* | 1.00 | 1.00 | 1.00 | 160 |
| *ledgerLine* | 1.00 | 0.76 | 0.86 | 5893 |
| *repeatDot* | 0.00 | 0.00 | 0.00 | 160 |
| *segno* | 1.00 | 1.00 | 1.00 | 11 |
| *coda* | 1.00 | 1.00 | 1.00 | 7 |
| *clefG* | 1.00 | 1.00 | 1.00 | 505 |
| *clefCAlto* | 0.55 | 1.00 | 0.71 | 34 |
| *clefCTenor* | 1.00 | 0.07 | 0.12 | 30 | 
| *clefF* | 1.00 | 1.00 | 1.00 | 302 |
| *clefUnpitchedPercussion* | 0.00 | 0.00 | 0.00 | 0 |
| *clef8* | 0.00 | 0.00 | 0.00 | 72 |
| *clef15* | 0.00 | 0.00 | 0.00 | 7 |
| *timeSig0* | 1.00 | 1.00 | 1.00 | 20 |
| *timeSig1* | 1.00 | 1.00 | 1.00 | 16 |
| *timeSig2* | 1.00 | 1.00 | 1.00 | 38 |
| *timeSig3* | 1.00 | 1.00 | 1.00 | 36 |
| *timeSig4* | 0.99 | 1.00 | 1.00 | 114 |
| *timeSig5* | 0.00 | 0.00 | 0.00 | 0 |
| *timeSig6* | 1.00 | 1.00 | 1.00 | 36 |
| *timeSig7* | 0.00 | 0.00 | 0.00 | 0 |
| *timeSig8* | 1.00 | 1.00 | 1.00 | 64 |
| *timeSig9* | 1.00 | 1.00 | 1.00 | 8 |
| *timeSigCommon* | 1.00 | 1.00 | 1.00 | 11 |
| *timeSigCutCommon* | 1.00 | 1.00 | 1.00 | 12 |
| *noteheadBlackOnLine* | 1.00 | 0.51 | 0.68 | 7742 |
| *noteheadBlackOnLineSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *noteheadBlackInSpace* | 0.44 | 1.00 | 0.62 | 7575 |
| *noteheadBlackInSpaceSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *noteheadHalfOnLine* | 1.00 | 0.55 | 0.71 | 571 |
| *noteheadHalfOnLineSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *noteheadHalfInSpace* | 1.00 | 0.53 | 0.69 | 610 |
| *noteheadHalfInSpaceSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *noteheadWholeOnLine* | 0.99 | 1.00 | 1.00 | 193 |
| *noteheadWholeOnLineSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *noteheadWholeInSpace* | 1.00 | 1.00 | 1.00 | 244 |
| *noteheadWholeInSpaceSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *noteheadDoubleWholeOnLine* | 0.50 | 1.00 | 0.67 | 3 |
| *noteheadDoubleWholeOnLineSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *noteheadDoubleWholeInSpace* | 0.00 | 0.00 | 0.00 | 3 |
| *noteheadDoubleWholeInSpaceSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *augmentationDot* | 0.00 | 0.00 | 0.00 | 1467 |
| *stem* | 0.90 | 1.00 | 0.95 | 14511 |
| *tremolo1* | 0.00 | 0.00 | 0.00 | 0 |
| *tremolo2* | 0.00 | 0.00 | 0.00 | 0 |
| *tremolo3* | 1.00 | 1.00 | 1.00 | 3 |
| *tremolo4* | 0.00 | 0.00 | 0.00 | 0 |
| *tremolo5* | 0.00 | 0.00 | 0.00 | 0 |
| *flag8thUp* | 1.00 | 0.72 | 0.84 | 317 |
| *flag8thUpSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *flag16thUp* | 1.00 | 1.00 | 1.00 | 43 |
| *flag32ndUp* | 0.00 | 0.00 | 0.00 | 6 |
| *flag64thUp* | 0.67 | 1.00 | 0.80 | 4 |
| *flag128thUp* | 0.55 | 1.00 | 0.71 | 6 |
| *flag8thDown* | 0.96 | 0.74 | 0.83 | 534 |
| *flag8thDownSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *flag16thDown* | 1.00 | 1.00 | 1.00 | 50 |
| *flag32ndDown* | 1.00 | 0.91 | 0.95 | 11 |
| *flag64thDown* | 0.75 | 0.38 | 0.50 | 8 |
| *flag128thDown* | 0.00 | 0.00 | 0.00 | 0 |
| *accidentalFlat* | 0.44 | 0.42 | 0.43 | 166 |
| *accidentalFlatSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *accidentalNatural* | 0.91 | 1.00 | 0.95 | 393 |
| *accidentalNaturalSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *accidentalSharp* | 0.58 | 0.62 | 0.60 | 512 |
| *accidentalSharpSmall* | 0.00 | 0.00 | 0.00 | 0 |
| *accidentalDoubleSharp* | 0.00 | 0.00 | 0.00 | 2 |
| *accidentalDoubleFlat* | 0.00 | 0.00 | 0.00 | 0 |
| *keyFlat* | 0.86 | 0.87 | 0.87 | 684 |
| *keyNatural* | 0.00 | 0.00 | 0.00 | 38 |
| *keySharp* | 0.68 | 0.64 | 0.66 | 637 |
| *articAccentAbove* | 0.81 | 0.97 | 0.88 | 116 |
| *articAccentBelow* | 0.83 | 0.33 | 0.47 | 46 |
| *articStaccatoAbove* | 0.00 | 0.00 | 0.00 | 332 |
| *articStaccatoBelow* | 0.00 | 0.00 | 0.00 | 204 |
| *articTenutoAbove* | 0.00 | 0.00 | 0.00 | 19 |
| *articTenutoBelow* | 0.00 | 0.00 | 0.00 | 6 |
| *articStaccatissimoAbove* | 0.00 | 0.00 | 0.00 | 6 |
| *articStaccatissimoBelow* | 0.00 | 0.00 | 0.00 | 4 |
| *articMarcatoAbove* | 0.00 | 0.00 | 0.00 | 48 |
| *articMarcatoBelow* | 0.00 | 0.00 | 0.00 | 27 |
| *fermataAbove* | 1.00 | 1.00 | 1.00 | 91 |
| *fermataBelow* | 1.00 | 1.00 | 1.00 | 49 |
| *caesura* | 1.00 | 0.16 | 0.27 | 58 |
| *restDoubleWhole* | 0.00 | 0.00 | 0.00 | 1 |
| *restWhole* | 0.97 | 0.19 | 0.32 | 494 |
| *restHalf* | 0.50 | 0.01 | 0.02 | 90 |
| *restQuarter* | 1.00 | 1.00 | 1.00 | 579 |
| *rest8th* | 1.00 | 0.99 | 1.00 | 530 |
| *rest16th* | 0.99 | 1.00 | 1.00 | 178 |
| *rest32nd* | 1.00 | 1.00 | 1.00 | 41 |
| *rest64th* | 1.00 | 1.00 | 1.00 | 18 |
| *rest128th* | 1.00 | 1.00 | 1.00 | 7 |
| *restHNr* | 0.00 | 0.00 | 0.00 | 0 |
| *dynamicP* | 1.00 | 0.83 | 0.91 | 266 |
| *dynamicM* | 1.00 | 0.29 | 0.45 | 79 |
| *dynamicF* | 1.00 | 1.00 | 1.00 | 329 |
| *dynamicS* | 1.00 | 0.53 | 0.70 | 43 |
| *dynamicZ* | 0.00 | 0.00 | 0.00 | 22 |
| *dynamicR* | 0.00 | 0.00 | 0.00 | 2 |
| *graceNoteAcciaccaturaStemUp* | 0.00 | 0.00 | 0.00 | 0 |
| *graceNoteAppoggiaturaStemUp* | 0.00 | 0.00 | 0.00 | 0 |
| *graceNoteAcciaccaturaStemDown* | 0.00 | 0.00 | 0.00 | 0 |
| *graceNoteAppoggiaturaStemDown* | 0.00 | 0.00 | 0.00 | 0 |
| *ornamentTrill* | 1.00 | 0.94 | 0.97 | 16 |
| *ornamentTurn* | 1.00 | 1.00 | 1.00 | 28 |
| *ornamentTurnInverted* | 1.00 | 1.00 | 1.00 | 5 |
| *ornamentMordent* | 1.00 | 1.00 | 1.00 | 37 |
| *stringsDownBow* | 0.00 | 0.00 | 0.00 | 6 |
| *stringsUpBow* | 1.00 | 1.00 | 1.00 | 28 |
| *arpeggiato* | 0.77 | 0.80 | 0.79 | 30 |
| *keyboardPedalPed* | 1.00 | 0.80 | 0.89 | 10 |
| *keyboardPedalUp* | 1.00 | 0.97 | 0.98 | 33 |
| *tuplet3* | 0.00 | 0.00 | 0.00 | 71 |
| *tuplet6* | 0.00 | 0.00 | 0.00 | 58 |
| *fingering0* | 0.00 | 0.00 | 0.00 | 62 |
| *fingering1* | 0.00 | 0.00 | 0.00 | 67 |
| *fingering2* | 0.00 | 0.00 | 0.00 | 35 |
| *fingering3* | 0.00 | 0.00 | 0.00 | 39 |
| *fingering4* | 0.00 | 0.00 | 0.00 | 31 |
| *fingering5* | 0.00 | 0.00 | 0.00 | 0 |
| *slur* | 0.99 | 0.97 | 0.98 | 652 |
| *beam* | 1.00 | 0.60 | 0.75 | 4170 |
| *tie* | 0.97 | 0.92 | 0.94 | 590 |
| *restHBar* | 1.00 | 0.96 | 0.98 | 27 |
| *dynamicCrescendoHairpin* | 1.00 | 1.00 | 1.00 | 58 |
| *dynamicDiminuendoHairpin* | 1.00 | 1.00 | 1.00 | 57 |
| *tuplet1* | 0.00 | 0.00 | 0.00 | 0 |
| *tuplet2* | 0.00 | 0.00 | 0.00 | 0 |
| *tuplet4* | 0.00 | 0.00 | 0.00 | 0 |
| *tuplet5* | 0.00 | 0.00 | 0.00 | 1 |
| *tuplet7* | 0.00 | 0.00 | 0.00 | 0 |
| *tuplet8* | 0.00 | 0.00 | 0.00 | 0 |
| *tuplet9* | 0.00 | 0.00 | 0.00 | 0 |
| *tupletBracket* | 0.78 | 0.93 | 0.85 | 15 |
| *staff* | 1.00 | 0.99 | 0.99 | 845 |
| *ottavaBracket* | 0.75 | 1.00 | 0.86 | 3 |
| | | | | |
| ***micro avg*** | 0.78 | 0.78 | 0.78 | 54528 |
| ***macro avg*** | 0.50 | 0.46 | 0.46 | 54528 |
| ***weighted avg((())) | 0.83 | 0.78 | 0.77 | 54528 |

