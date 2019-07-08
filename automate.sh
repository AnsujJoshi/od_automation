# # --> convert images to tfrecord
# # --> run CMLE job and save all the outputs in dvc

echo '#############################################################'
echo '######################## BASH ################################'
echo '#############################################################'
source ./gcp.config

# # echo $INPUT_BUCKET
python dataflow.py --bucket=$INPUT_BUCKET \
 --image_folder_path=$IMAGE_INPUT \
 --label_folder_path=$LABEL_INPUT \
 --output_dir=$OUTPUT \
 --runner='DataflowRunner' \
 --project=$PROJECT\
 --jobname=$DATAFLOW_JOBNAME

echo 'PLEASE CHECK DATAFLOW SERVICE'

### --> Change config file according to the output dir in dataflow
### --> Write code to check if tfrecord is present or not
### --> Shift config file to gs
echo '#############################################################'
echo '################## CONFIG FILE ##############################'
echo '#############################################################'

gsutil cp -r config.config gs://$OUTPUT_BUCKET/data/


echo 'PLEASE CHECK THAT CONFIG FILE IS PRESENT'

echo '#############################################################'
echo '##################### TRAINING ##############################'
echo '#############################################################'

gsutil -q stat gs://$INPUT_BUCKET/$OUTPUT/output_tfrecord.record-00000-of-00001
while :
do
    if [ $?==0 ]
    then
            gcloud ml-engine jobs submit training $CMLE_TRAIN_JOBNAME \
                --runtime-version 1.12 \
                --job-dir=gs://$OUTPUT_BUCKET/$JOB_DIR/$UID \
                --packages models/research/dist/object_detection-0.1.tar.gz,models/research/slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
                --module-name object_detection.model_tpu_main \
                --scale-tier BASIC_TPU \
                --region us-central1 \
                -- \
                --tpu_zone us-central1 \
                --model_dir=gs://$OUTPUT_BUCKET/$MODEL_DIR/$UID \
                --pipeline_config_path=gs://$OUTPUT_BUCKET/data/config.config

            gcloud ml-engine jobs submit training $CMLE_EVAL_JOBNAME\
                --runtime-version 1.12 \
                --job-dir=gs://$OUTPUT_BUCKET/$JOB_DIR/$UID \
                --packages models/research/dist/object_detection-0.1.tar.gz,models/research/slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
                --module-name object_detection.model_main \
                --region us-central1 \
                --scale-tier BASIC_GPU \
                -- \
                --model_dir=gs://$OUTPUT_BUCKET/$MODEL_DIR/$UID \
                --pipeline_config_path=gs://$OUTPUT_BUCKET/data/config.config \
                --checkpoint_dir=gs://$OUTPUT_BUCKET/$MODEL_DIR/$UID
            break
    fi
done
echo 'PLEASE CHECK THAT CMLE JOB IS RUNNING'

echo '#############################################################'
echo '####################### VERSIONING ##########################'
echo '#############################################################'

python check_status.py --jobname=$CMLE_TRAIN_JOBNAME --projectname=$PROJECT --uid=$UID
    