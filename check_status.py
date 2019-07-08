'''Gets status of the job'''

from googleapiclient import discovery
import os
import argparse



if __name__=='__main__':

    ml = discovery.build('ml', 'v1')

    parser = argparse.ArgumentParser()

    parser.add_argument('--uid', 
    help = 'uid to be given',
    required=True)

    parser.add_argument('--jobname', 
    help = 'name of the job',
    required=True)

    parser.add_argument('--projectname', 
    help = 'name of the project',
    required=True)

    arguments = parser.parse_args()
    args = arguments.__dict__

    # jobName = str(args['jobname'])
    # projectName = str(args['projectname'])
    UID = str(args['uid'])

    projectName = 'di-safetyfop-us-poc-1'
    projectId = 'projects/{}'.format(projectName)
    jobName = 'object_detection_eval_07_05_2019_22_37_45'
    jobId = '{}/jobs/{}'.format(projectId, jobName)

    request = ml.projects().jobs().get(name=jobId)
    i = 0 
    while True:
        response = request.execute()
        
        if response['state']=='SUCCEEDED':
            break
    