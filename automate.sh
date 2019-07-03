# # --> convert images to tfrecord
# # --> run CMLE job and save all the outputs in dvc


source ./gcp.config

python dataflow.py --bucket=$INPUT_BUCKET \
 --image_folder_path=$IMAGE_INPUT \
 --label_folder _path=$LABEL_INPUT \
 --output_dir=$OUTPUT \
 --runner='DataflowRunner' \
 --project=$PROJECT\
 --jobname=$JOBNAME


### --> Change config file according to the output dir in dataflow
### --> Write code to check if tfrecord is present or not
### --> Shift config file to gs

# gsutil cp -r $LOCAL_CONFIG_FILE gs://$BUCKET/data/

### TPU training    
gcloud ml-engine jobs submit training `whoami`_tpu_object_detection_disneyds_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.12 \
    --job-dir=gs://$OUTPUT_BUCKET/$JOB_DIR \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,./pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_tpu_main \
    --scale-tier BASIC_TPU \
    --region us-central1 \
    -- \
    --tpu_zone us-central1 \
    --model_dir=gs://$OUTPUT_BUCKET/$MODEL_DIR \
    --pipeline_config_path=gs://$OUTPUT_BUCKET/data/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_tpu_cw_pascal5.config

### GPU Evaluation
gcloud ml-engine jobs submit training object_detection_eval_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.12 \
    --job-dir=gs://$OUTPUT_BUCKET}/$JOB_DIR \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/home/mohsin/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --model_dir=gs://$OUTPUT_BUCKET/$MODEL_DIR \
    --pipeline_config_path=gs://$OUTPUT_BUCKET/data/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_tpu_cw_pascal5.config \
    --checkpoint_dir=gs://$OUTPUT_BUCKET/$MODEL_DIR

# tensorboard --logdir=gs://$OUTPUT_BUCKET/$MODEL_DIR