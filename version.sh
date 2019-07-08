#Versions the data and the training

# use this file with command *** bash version.sh {UID} ***

source gcp.config

git checkout dev

git checkout -b $UID

bash automate.sh

gsutil -m cp -r gs://$OUTPUT_BUCKET/$MODEL_DIR .

dvc add -f $UID.dvc $MODEL_DIR

rm -r $MODEL_DIR

git add $UID.dvc
 
git commit -m $UID 'created, please verify'

git tag -a $UID -m 'dvc version created with '$1

echo "please check your branch"

git push origin $UID
