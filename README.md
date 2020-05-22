# OD Automation scripts
## Folder Structure
```
--> automate.sh
--> version.sh 
--> new_system.sh
--> visualize.sh
--> dataflow.py
--> check_status.py
--> gcp.config.tmp(needs to be edited before use)
--> config.config(needs to be edited before use)
*--> models --> research --> sdist
*--> models --> research --> slim
*--> models --> research--> ..
```
###### *models folder come after you run new_system.sh*
---
----------------------------
## Model Creation process

### Tagging

LabelImg, is a graphical image annotation tool for creating bounding boxes in images.

This tool is used by taggers to create bounding boxes on the images.
The coordinates of these bounding boxes are stored in the xml files which are later given the following name *image_name*.xml
After that the tagged images with their tagged label are given out and stored in a google cloud storage bucket.


*Insert Labelimg tagged image*


### Conversion to tfrecord

The tagged images and their labels are then converted into a tfrecord, because the model to be trained needs to be in a readable format for the Tensorflow Object detection code.
To convert this model into tfrecord a Dataflow pipeline is used which reads the data from the GCS and stores back a tfrecord in the GCS bucket of your choice. The following code is used.

Training
For training the model, tensorflow object detection model is used.
The training is done on a TPU for faster training, and evaluation is done on GPUs to save money.
The training code first triggers a job on training job on CMLE and simultaneously a validation job is also triggered to keep a check on the metrics.
After the training is complete we get the checkpoints which are again stored in the GCS.
---
---
## How to run this code

###### *IMPORTANT : Fill in appropriate values in gcp.config and config.config. After filling those files follow the next steps.*
#### New System

 
    pip install requirements.txt
    
    bash new_system.sh

    bash version.sh

#### Old system


    bash version.sh


##### *** Please fill in the gcp.config.tmp and convert it to gcp.config before running the code ***
---
---
