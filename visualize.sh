source gcp.config

tensorboard --logdir=gs://$OUTPUT_BUCKET/$MODEL_DIR
