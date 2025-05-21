# Project Overview
This repository contains the code developed for the bachelor's thesis  
*Evaluation of Computer Vision Methods for Building Detection in Orthophotos"  
by Viktorija Aužele (University of Latvia 2025).


The aim of this study is to evaluate the effectiveness of the computer vision algorithms U-Net, DeepLabv3+, YOLOv8n-seg, and YOLOv9c-seg in building recognition from orthophotos and to assess their practical applicability in construction monitoring. Fragments from the 6th-cycle color orthophoto map produced by the Latvian Geospatial Information Agency were used to train the algorithms using data from the Salaspils area and to test the resulting models on data from the Ogre area. Model effectiveness was assessed at the pixel-level by comparing the predicted masks with manually prepared reference masks and by calculating performance metrics, including precision, accuracy, recall, the F1 score, and the Intersection over Union (IoU) coefficient. To assess the practical applicability of the models in construction monitoring, the pixel-level overlap between the predicted masks and cadastral building polygons was analyzed.
The results indicate that U-Net and DeepLabv3+ achieved higher segmentation accuracy and demonstrated greater potential for practical implementation.



The code used in this study is divided into separate scripts for each segmentation model and a set of helper scripts used for data preparation, result processing, and evaluation. This structure helps to keep the workflow organized and supports reproducibility.


## 📁 /YOLO/

This directory contains scripts related to the YOLOv8n-seg and YOLOv9c-seg models used for building segmentation in orthophoto images.

**Contents:**
- `data.yaml` – configuration file specifying class labels, paths, and other parameters.

**Subdirectory:** /YOLO/scripts/
- `yolov8n-train.py` – training script for the YOLOv8n-seg model.
- `yolov9c-train.py` – training script for the YOLOv9c-seg model.
- `test.py – script` for evaluating the trained YOLO models on the test dataset.



## 📂 /SMP_UNET/

This folder includes the implementation of the U-Net segmentation model using the segmentation_models_pytorch library.

**Contents:**
- `train.py` – training script for the U-Net model.
- `test.py` – evaluation script for testing the trained model.
- `dataset.py` – dataset class for loading and preprocessing data for the model.



## 📂 /SMP_DEEPLABV3/

This folder includes the implementation of the DeepLabv3+ segmentation model using the segmentation_models_pytorch library.

**Contents:**
- `train.py` – training script for the DeepLabv3+ model.
- `test.py` – script for evaluating segmentation results.
- `dataset.py` – dataset handling and preprocessing logic for model input.



## 📂 /OTHER_SCRIPTS/

This folder includes helper scripts that are not model-specific but are essential for data processing, mask reconstruction, and result evaluation.

**Contents:**
- `results.py` – calculates evaluation metrics such as precision, accuracy, recall, F1 score, and Intersection over Union (IoU) by comparing predicted masks to ground truth data.
- `test_result_reconstruction.py` – merges individual tile-based segmentation outputs back into a single raster file.
- `tiles.py` – script for tiling large orthophoto images into smaller fragments suitable for model input.
- `txt_labels.py` – handles class label formatting, possibly for YOLO annotations or label files.
