# EPIC-KITCHENS VISOR (Video Segmentations and Object Relations) Dataset
Release Date: Aug 2022

## Authors
Ahmad Darkhalil* (1)
Dandan Shan* (2)
Bin Zhu* (1)
Jian Ma* (1)
Amlan Kar (3)
Richard Higgins (2)
Sanja Fidler (3)
David Fouhey (2)
Dima Damen (1)

1: University of Bristol, United Kingdom
2: University of Michigan, United States
3: University of Toronto, Canada


## Citing
When using VISOR annotations, kindly reference both the annotations paper (VISOR) and the EPIC-KITCHENS-100 paper as follows:

- Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar, Richard Higgins, Sanja Fidler, David Fouhey, Dima Damen (2022). EPIC-KITCHENS VISOR Benchmark
VIdeo Segmentations and Object Relations. Early Access OpenReview.

- Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, Michael Wray (2022). Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100. International Journal of Computer Vision (IJCV) vol 130, pp 33-55.

## Ownership and License

All annotations are owned by the University of Bristol under the research license agreement 2021 - 3107, signed by all parties on Jan 2022.

All files in this dataset are copyright by us and published under the Creative Commons Attribution-NonCommerial 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc/4.0/). This means that you must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. You may not use the material for commercial purposes.

For commercial licenses, contact the University of Bristol at: uob-epic-kitchens@bristol.ac.uk 

## Dataset Details

This deposit contains manually collected pixel-level segmentations for a subset of videos from the dataset EPIC-KITCHENS-100. These represent 30 recorded hours (train/val) and 6 hours of test (only images released).

This README contains information about annotations format.  Please see [epic-kitchens VISOR website](https://github.com/epic-kitchens/VISOR) for additional details, open challenges and the latest starter code.


## Videos and RGB Images

The original videos can be downloaded from two separate DOIs:
http://dx.doi.org/10.5523/bris.3h91syskeag572hl6tvuovwv4d 
and
http://dx.doi.org/10.5523/bris.2g1n6qdydwa9u22shpxqzp0t8m

However, we also release sparse annotation frames for all splits of VISOR (train/val/test) with the following structure:
sparse
-- train
---- P01
-------P01_01.zip
-- val
-- test

We provide a mapping from these frames to the originally released rgb_frames in EPIC-KITCHENS. This avoids any differences in the frame extraction if any.

## Ground Truth - Sparse Annotations

As explained in Sec 2.1, we provide manually annotated sparse segmentations, at the rate of (roughly) 2 frames per action clip. We use JSON formatting annotations, one per video, with the following fields
*image*
-- image_path: the folder and file name in the VISOR file structure. Note that this might not exactly match the EPIC-KITCHENS frame numbers (see Sec frame extraction for correspondences)
-- name: image filename
-- subsequence: Refer to Sec 2.1 where we define subsequences within videos with consistent set of entities. This field refers to the sequence number and is used for the VOS benchmark
-- video: video filename - matches EPIC-KITCHENS-100 video file name
*annotations* [one per mask]
-- id: Unique mask ID
-- name: open vocabulary entity name
-- class_id: key for EPIC-KITCHENS class number reflecting closed vocabulary entity name (see Sec classes for correspondences)
-- exhaustive: flag (y/n) to indicate whether all instances of the entity have been segmented. When 'n' is selected, background non-active instances of the same entity are present. We use 'inconclusive' when a consensus amongst annotators could not be found
-- in_contact_object: [only for hand and on-hand gloves] this field offers one of several options: the ID of the object that the hand is in contact with in this frame, or hand-not-in-contact when the hand is not touching any mask, none-of-the-above when the hand is touching an object that is not segmented, or inconclusive when a consensus amongst annotators could not be found.
-- on_which_hand: [only for gloves] indicating whether the glove is worn on a hand and which hand side is it. 
-- segments: polygon-based representation of each mask

## Interpolations - Dense Annotations

As explained in Sec 2.4, two consecutive ground truth segmentations are used to interpolate intermediate masks. The interpolations are filtered and only high J&F scored interpolations are provided.

In addition to the flags above, additional flags are available in interpolations:

-- type: 1: start/end ground-truth frames, that have been filtered to only include entities that are present at both the start and end frames. These are replicated here. 0: interpolated frames
-- interpolation: unique incremental ID for each interpolation in the dataset

## Frame Extraction (frame_mapping.json)

Note that we had to re-extract frames from the dataset due to a change in ffmpeg versions. We provide a file that maps frame numbers between VISOR and publicly released rgb_frames in EPIC-KITCHENS-55 and EPIC-KITCHENS-100.
You can find this at: frame_mapping.json

## Classes (EPIC_100_noun_classes_v2.csv)

As new open-vocabulary entities have been added with the VISOR segmentations, we provide an updated noun classes for EPIC-KITCHENS.
These include all new entities, mapped to the EPIC-KITCHENS-100 classes.
We also add 4 additional classes: left hand, right hand, left glove, right glove - as prior classes didn't separate hand or glove sides. Note that we do not remove the original hand and glove classes for consistency with other benchmarks.

