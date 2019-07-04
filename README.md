
# OD Automation scripts
## Folder Structure

 --> automate.sh \
--> version.sh \
--> new_system.sh\
--> gcp.config.tmp(needs to be edited before use)\
--> config.config(needs to be edited before use)\
*--> models --> research --> sdist \
*--> models --> research --> slim\
*--> models --> research--> ..

#### *models folder come after you run new_system.sh* 
## How to run this code

#### New System

### *IMPORTANT : Fill in appropriate values in gcp.config and config.config. After filling those files follow the next steps.*


`
pip install requirements.txt
`

`
bash new_system.sh
`

`
bash automate.sh
`
#### Old system

`
bash automate.sh
`

*** Please fill in the gcp.config.tmp and convert it to gcp.config before running the code ***