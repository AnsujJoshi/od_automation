
'''Imports images from the folder, augment them and convert into tfrecord'''

import tensorflow as tf
import argparse
# import zipline
import os
import uuid
import glob
import logging

from google.cloud import storage

import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions
# from xml_convertor import start

# ROOT_DIR = "."
# IMAGES_DIR = "images"
# ANNOTATIONS_DIR_PREFIX = 'all_tagged_data/labels'
# DESTINATION_DIR = "labels_xml2"


def upload_to_gcs(project, bucket, gcs_path, filepath):
    '''Uploads a file to gcs bucket
    Args:
        project : str, project id
        bucket : str, bucket name
        gcs_path :  str, gcs path after bucket name, if any
        filepath : str, path of the file to upload'''
    storage_client = storage.Client(project=project)

    bucket = storage_client.get_bucket(bucket)

    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(filepath)


def download_from_gcs(project, bucket, gcs_path, filepath):
    '''Downloads a file from gcs bucket
    Args:
        project : str, project id
        bucket : str, bucket name
        gcs_path :  str, gcs path after bucket name, if any
        filepath : str, path of the file to download'''
    storage_client = storage.Client(project=project)

    bucket = storage_client.get_bucket(bucket)

    blob = bucket.blob(gcs_path+'/*')
    blob.download_to_filename(filepath)


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

    # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def __del__(self):
        self._sess.close()


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label_list,
                        height, width):
    """Build an Example proto for an example.

    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label_int: integer, identifier for ground truth (0-based)
    label_str: string, identifier for ground truth, e.g., 'daisy'
    height: integer, image height in pixels
    width: integer, image width in pixels
    Returns:
    Example proto
    """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    labels = []
    x1 = []
    y1 = []
    cx1 = []
    cy1 = []
    for i in range(len(label_list)):
        label = int(label_list[i]['class_id'])
        temp_x1 = float(label_list[i]['x1'])
        temp_y1 = float(label_list[i]['y1'])
        temp_cx1 = float(label_list[i]['cx1'])
        temp_cy1 = float(label_list[i]['cy1'])
        labels.append(label)
        x1.append(temp_x1)
        y1.append(temp_y1)
        cx1.append(temp_cx1)
        cy1.append(temp_cy1)

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/colorspace': _bytes_feature(colorspace),
                'image/channels': _int64_feature(channels),
                'image/class/label': _int64_feature(labels),
                'image/class/x1': _float_feature(x1),
                'image/class/y1': _float_feature(y1),
                'image/class/cx1': _float_feature(cx1),
                'image/class/cy1': _float_feature(cy1),
                'image/format': _bytes_feature(image_format),
                'image/filename': _bytes_feature(os.path.basename(filename)),
                'image/encoded': _bytes_feature(image_buffer)
            }))
    return example


def _convert_txt_to_label_dict(txtpath):
    '''Converts text file into label dict to be storedas a feature into tfrecord
    Args:
        txtpath: str, path of the textfile

    Returns:
        dic_list : list, list of dictionary containing the label 
        eg : [{'class_id': '4',
                'x1': '0.603125',
                'y1': '0.367130',
                'cx1': '0.147917',
                'cy1': '0.269444'},
                {'class_id': '4',
                'x1': '0.863021',
                'y1': '0.143981',
                'cx1': '0.075000',
                'cy1': '0.165741'}]
    '''
    f = open(txtpath)
    lines = f.readlines()
    # print(root)
    dic_list = []
    for l in lines:
        l = l.replace('\n', '')
        elems = l.split(" ")
        d = dict()
        d['class_id'] = elems[0]
        d['x1'] = elems[1]
        d['y1'] = elems[2]
        d['cx1'] = elems[3]
        d['cy1'] = elems[4]
        dic_list.append(d)
    return dic_list


def _get_image_data(filename, coder):
    """Process a single image file.

    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as ifp:
        image_data = ifp.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def convert_to_example(csvline):
    """Parse a line of CSV file and convert to TF Record.

    Args:
    filename: name of the file
    Yields:
    serialized TF example if the label is in categories
    """
    filename, labelpath = csvline.encode('ascii', 'ignore').split(',')

    label_list = _convert_txt_to_label_dict(labelpath)

    coder = ImageCoder()
    image_buffer, height, width = _get_image_data(filename, coder)
    del coder
    example = _convert_to_example(filename, image_buffer, label_list,
                                  height, width)
    print(example.SerializeToString())
    logging.info(example)
    yield example.SerializeToString()


def create_dirs(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_folder_path', 
    help = 'Path of the folder containing the images',
    required=True)

    parser.add_argument('--label_folder_path',
    help = 'Path of the folder containing the labels in txt format',
    required=True)

    parser.add_argument('--output_dir', 
    help= 'Path of the output directory to save the TFRecords', 
    required=True)

    parser.add_argument('--runner',
    help = 'Runner to the run the dataflow pipeline',
    required=True)

    parser.add_argument('--staging_bucket',
    help = 'Runner to the run the dataflow pipeline',
    required=False)

    parser.add_argument('--temp_location',
    help = 'Runner to the run the dataflow pipeline',
    required=False)

    parser.add_argument('--project',
    help = 'Runner to the run the dataflow pipeline',
    required=True)


    arguments = parser.parse_args()
    print(arguments)

    args = arguments.__dict__


    RUNNER = str(args['runner'])
    OUTPUTDIR = str(args['output_dir'])
    PROJECT = str(args['project'])
    JOBNAME = 'betatesting1'

    # if str(args['staging_bucket']) =None:
    staging_bucket = OUTPUTDIR
    # else:
    #     staging_bucket = args['staging_bucket']
    
    # if str(args['temp_location']) == None:
    temp_folder = OUTPUTDIR
    # else:
    #     temp_folder = args['staging_bucket']


    print(staging_bucket, temp_folder)
    options = PipelineOptions()

    # # For Cloud execution, set the Cloud Platform project, job_name,
    # # staging location, temp_location and specify DataflowRunner.
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = PROJECT
    google_cloud_options.job_name = JOBNAME
    google_cloud_options.staging_location = staging_bucket
    google_cloud_options.temp_location = temp_folder
    options.view_as(StandardOptions).runner = RUNNER


    print(str(args['image_folder_path']))
    print(type(str(args['image_folder_path'])))

    IMAGE_BUCKET = str(args['image_folder_path']).split('/')[2]
    download_image_path = str(args['image_folder_path']).replace('gs://{}/'.format(IMAGE_BUCKET), '')
    download_from_gcs(PROJECT, IMAGE_BUCKET, download_image_path, './images')

    LABEL_BUCKET = str(args['label_folder_path']).split('/')[2]
    download_label_path = str(args['label_folder_path']).replace('gs://{}/'.format(LABEL_BUCKET), '')
    download_from_gcs(PROJECT, LABEL_BUCKET, download_label_path, './labels')
    
    os.system('mkdir images')
    os.system('mkdir labels')
    os.system('gsutil -m cp -r {} ./'.format(str(args['image_folder_path'])))
    os.system('gsutil -m cp -r {} ./'.format(str(args['label_folder_path'])))

    image_path = './images' # Don't put a / after the path, Thanks
    label_path = './labels'
    image_path = '{}/*.jpg'.format(image_path)
    image_list = glob.glob(image_path)


    with open('temp.csv', 'w') as csvfile:
        for i in range(len(image_list)):
            raw_name = image_list[i].split('/')[-1][:-4]
            csvfile.write('{},{}\n'.format(
                image_list[i], '{}/{}.txt'.format(label_path, raw_name)))
    print('TEMP FILE CREATED')

    create_dirs(OUTPUTDIR)
    print('DIRECTORY CREATED')

    with beam.Pipeline(options=options) as p:
        _ = (
            p |
            'Read data from local' >> beam.io.ReadFromText('temp.csv') |
            'Convert data to tfrecord' >> beam.FlatMap(lambda line: convert_to_example(line)) |
            'Save tfrecords' >> beam.io.tfrecordio.WriteToTFRecord(OUTPUTDIR, num_shards=1)
        )
    
    # OUT_BUCKET = OUTPUTDIR.split('/')[2]
    # upload_file_path = OUTPUTDIR.replace('gs://{}/'.format(OUT_BUCKET), '')
    # upload_to_gcs(PROJECT, OUT_BUCKET, upload_file_path, './tf_out*')
