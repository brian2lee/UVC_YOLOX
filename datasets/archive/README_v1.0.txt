*******************************************
* MVTec Screws V1.0                       *
*                                         *
* Author: MVTec Software GmbH, July 2020. *
*         https://www.mvtec.com           *
*******************************************

All files are as in the MVTec Screws example dataset for oriented object detection, released with
HALCON version 19.05. The state of the dataset and images is as of release version 20.05.

***********
* License *
***********

The dataset, i.e. the images and the annotations, are licensed under the creative commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. See
https://creativecommons.org/licenses/by-nc-sa/4.0/ for more information.
For using the data in a way that falls under the commercial use clause of the license,
please contact us.

***************
* Attribution *
***************

If you use this dataset in scientific work, please cite our paper:

Markus Ulrich, Patrick Follmann, Jan-Hendrik Neudeck: 
A comparison of shape-based matching with deep-learning-based object detection;
in: Technisches Messen, 2019, DOI 10.1515/teme-2019-0076.

************
* Content: *
************

MVTec Screws contains 384 images of 13 different types of screws and nuts on a wooden background.
The objects are labeled by oriented bounding boxes and their respective category. Overall, there
are 4426 of such annotations.
The exemplary splits are those that have been used in the above mentioned publication. Initially,
they have been selected randomly, such that approximately 70% of the instances of each category are
within the training split, and 15% each in the validation and test splits.

* folder images contains the screw images.
* mvtec_screws.json contains the annotations for all images in COCO format.
* mvtec_screws_train/val/test.json contain examplary splits as mentioned above, in COCO format.
* mvtec_screws.hdict contains the DLDataset unsplitted.
* mvtec_screws_split.hdict contains the DLDataset with splits.

******************************
* Usage of DLDataset-format: *
******************************

The .hdict files can be used within HALCON by reading them, e.g. via

read_dict (<path_to_mvtec_screws.hdict>, [], [], DLDataset)

The image path has to be set to the location of the images folder <path_to_images_folder> by

set_dict_tuple(DLDataset, 'image_dir', <path_to_images_folder>)

To store this information within the dataset, the dataset should be written by
write_dict (DLDataset, <path_to_mvtec_screws.hdict>, [], [])

In HALCON object detection we use subpixel-precise annotations with a pixel-centered coordinate-system, i.e.
the center of the top-left corner of the image is at (0.0, 0.0), while the top-left corner of the image is
located at (-.5, -.5). Note that when used within HALCON the dataset does not need to be converted, as this
format is also used within the deep learning based object detection of HALCON.

***************
* COCO Format *
***************

MVTec screws is a dataset for oriented box detection. We use a format that is very similar to that of the
COCO dataset (cocodataset.org). However, we need 5 parameters per box annotation to store the orientation.
We use the following labels.

Each box contains 5 parameters (row, col, width, height, phi), where

* 'row' is the subpixel-precise center row (vertical axis of the coordinate system) of the box.
* 'col' is the subpixel-precise center column (horizontal axis of the coordinate system) of the box.
* 'width' is the subpixel-precise width of the box. I.e. the length of the box parallel to the orientation 
  of the box.
* 'height' is the subpixel-precise width of the box. I.e. the length of the box perpendicular to the
  orientation of the box.
* 'phi' is the orientation of the box in radian, given in a mathematically positive sense and with respect
  to the horizontal (column) image axis. E.g. for phi = 0.0 the box is oriented towards the right side of 
  the image, for phi = pi/2 towards the top, for phi = pi towards the left, and for phi=-pi/2 towards the 
  bottom. Phi is always in the range (-pi, pi].

Note that width and height are defined in contrast to the DLDataset format in HALCON, where we use 
semi-axis lengths.

Coordinate system: In comparison to the pixel-centered coordinate-system of HALCON mentioned above,
                   for COCO it is common to set the origin to the top-left-corner of the top-left
				   pixel, hence in comparison to the DLDataset-format, (row,col) are shifted by (.5, .5).