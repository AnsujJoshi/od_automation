'''Dataflow pipeline'''


from google.cloud import storage
import argparse
import tensorflow as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import hashlib
import logging
from PIL import Image

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import WorkerOptions

def get_blob(bucket_name, prefix):

    # Instantiates a client
    storage_client = storage.Client()

    # Get GCS bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Get blobs in specific subirectory
    images_blobs = list(bucket.list_blobs(prefix=prefix))
    return images_blobs

  

class dataset_util():
    """Utility functions for creating TFRecord data sets."""

    def int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def int64_list_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


    def bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def bytes_list_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


    def float_list_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))


label2idx = {
    "led_1":1,
    "led_2":2,
    "led_3":3,
    "person":4,
    "person_zone3":5,
    "backrest":6,
    "door":7,
    "ride":8,
    "screen":9  
}

class_weights_dict = {
"led_1":24.578,
"led_2":65.9,
"led_3":2.133,
"person":0.413,
"person_zone3":2.835,
"backrest":0.499,
"door":8.160,
"ride":0.334,
"screen":1.696
}

class ConvertTOExample(beam.DoFn):

  def __init__(self):
    return None
  
  def int64_feature(self, value):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


  def int64_list_feature(self, value):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


  def bytes_feature(self, value):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


  def bytes_list_feature(self, value):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


  def float_list_feature(self, value):
    import tensorflow as tf
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  
  def _load_image(self, image_file):

    import tensorflow as tf
    import numpy as np
    # import cv2
    """Load single image from file.

    Args:
      image_file: RAW image file.
      byteswap: Swap byte ordering.

    Returns:
      Vector containing image intensity pixel values as numpy.array.
    """

    if image_file.startswith('gs://'):
      with tf.gfile.Open(image_file, 'rb') as f:
        img_str = f.read()
    else:
      with tf.gfile.Open(image_file, 'rb') as f:
        img_str = f.read()
    # nparr = np.fromstring(img_str, np.uint8)
    # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # height = img_np.shape[0]
    # width = img_np.shape[1]
    return img_str

  def _convert_to_example(self, filename, xml_file, use_class_weights=False):
      import tensorflow as tf
      import xml.etree.ElementTree as ET
      import logging
      import hashlib
      import os
      import io
      from PIL import Image

      idx2label = {
        1:"led_1",
        2:"led_2",
        3:"led_3",
        4:"person",
        5:"person_zone3",
        6:"backrest",
        7:"door",
        8:"ride",
        9:"screen"
        }
      label2idx = {v:k for k,v in idx2label.items()}
      with tf.gfile.Open(xml_file, 'rb') as f:
          file_data = f.read()
          root = ET.fromstring(file_data)

      image_name = root.find('filename').text
      print('Image_name', image_name)
      file_name = image_name.encode('utf8')
      size=root.find('size')
      width = int(size[0].text)
      height = int(size[1].text)
      xmin = []
      ymin = []
      xmax = []
      ymax = []
      classes = []
      classes_text = []
      truncated = []
      poses = []
      difficult_obj = []
      weights = []
      for member in root.findall('object'):
          label = member.find('name').text
          classes_text.append(str(label).encode('utf8'))
          print('LABEL', label)
          
          if use_class_weights:
              weights.append(class_weights_dict[str(label)])
          
          bbox = member.find('bndbox')
          xmin.append(float(bbox.find('xmin').text) / width)
          ymin.append(float(bbox.find('ymin').text) / height)
          xmax.append(float(bbox.find('xmax').text) / width)
          ymax.append(float(bbox.find('ymax').text) / height)
          difficult_obj.append(0)
          #if you have more than one classes in dataset you can change the next line
          #to read the class from the xml file and change the class label into its 
          #corresponding integer number, u can use next function structure
          try:
            classes.append(label2idx[label])   # i wrote 1 because i have only one class(person)
          except:
            print(label)
            return None
          truncated.append(0)
          poses.append('Unspecified'.encode('utf8'))
      # full_path = os.path.join("./"+img_and_anno_path+"/images", "{}".format(image_name))  #provide the path of images directory
      with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_jpg = fid.read()
      encoded_jpg_io = io.BytesIO(encoded_jpg)
      image = Image.open(encoded_jpg_io)
      if image.format != 'JPEG':
          raise ValueError('Image format not JPEG')
      # encoded_jpg = image_buffer

      key = hashlib.sha256(encoded_jpg).hexdigest()
      print('######## HERE #######')
      print(xmin, xmax, ymin, ymax)
      #create TFRecord Example
      
      if use_class_weights:
          example = tf.train.Example(features=tf.train.Features(feature={
              'image/height': self.int64_feature(height),
              'image/width': self.int64_feature(width),
              'image/filename': self.bytes_feature(file_name),
              'image/source_id': self.bytes_feature(file_name),
              'image/key/sha256': self.bytes_feature(key.encode('utf8')),
              'image/encoded': self.bytes_feature(encoded_jpg),
              'image/format': self.bytes_feature('jpeg'.encode('utf8')),
              'image/object/bbox/xmin': self.float_list_feature(xmin),
              'image/object/bbox/xmax': self.float_list_feature(xmax),
              'image/object/bbox/ymin': self.float_list_feature(ymin),
              'image/object/bbox/ymax': self.float_list_feature(ymax),
              'image/object/class/text': self.bytes_list_feature(classes_text),
              'image/object/class/label': self.int64_list_feature(classes),
              'image/object/difficult': self.int64_list_feature(difficult_obj),
              'image/object/truncated': self.int64_list_feature(truncated),
              'image/object/view': self.bytes_list_feature(poses),
              'image/object/weight': self.float_list_feature(weights)
          }))	
          return example
      else:
          example = tf.train.Example(features=tf.train.Features(feature={
              'image/height': self.int64_feature(height),
              'image/width': self.int64_feature(width),
              'image/filename': self.bytes_feature(file_name),
              'image/source_id': self.bytes_feature(file_name),
              'image/key/sha256': self.bytes_feature(key.encode('utf8')),
              'image/encoded': self.bytes_feature(encoded_jpg),
              'image/format': self.bytes_feature('jpeg'.encode('utf8')),
              'image/object/bbox/xmin': self.float_list_feature(xmin),
              'image/object/bbox/xmax': self.float_list_feature(xmax),
              'image/object/bbox/ymin': self.float_list_feature(ymin),
              'image/object/bbox/ymax': self.float_list_feature(ymax),
              'image/object/class/text': self.bytes_list_feature(classes_text),
              'image/object/class/label': self.int64_list_feature(classes),
              'image/object/difficult': self.int64_list_feature(difficult_obj),
              'image/object/truncated': self.int64_list_feature(truncated),
              'image/object/view': self.bytes_list_feature(poses)
          }))	
          return example

  def process(self, csvline):
    """Parse a line of CSV file and convert to TF Record.

    Args:
    filename: name of the file
    Yields:
    serialized TF example if the label is in categories
    """
    import logging

    filename, xml_file = csvline.split(',')
    logging.getLogger().setLevel(logging.INFO)
    # label_list = _convert_txt_to_label_dict(labelpath)
    print('####CNVT TO EXAMPLE #####')
    logging.info('####CNVT TO EXAMPLE #####')
    # image_string = self._load_image(filename)

    example = self._convert_to_example(filename, xml_file)
    print('Example', example)
    logging.info(" Example %s",example)
    yield example.SerializeToString()

if __name__ == "__main__":
    
    # bucket_name = 'testvideo_bucket_2'
    # prefix='test_data/images'

    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket', 
    help = 'name of the bucket cntaining the folders',
    required=True)

    parser.add_argument('--image_folder_path', 
    help = 'Path of the folder containing the images',
    required=True)

    parser.add_argument('--label_folder_path',
    help = 'Path of the folder containing the labels in xml format',
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

    parser.add_argument('--jobname',
    help = 'Runner to the run the dataflow pipeline',
    required=True)

    arguments = parser.parse_args()
    # print(arguments)

    args = arguments.__dict__
    print(args)
    BUCKET = str(args['bucket'])
    IMAGE_FOLDER_PATH = str(args['image_folder_path'])
    LABEL_FOLDER_PATH = str(args['label_folder_path'])
    RUNNER = str(args['runner'])
    if RUNNER=='DataflowRunner':
      OUTPUTDIR = 'gs://'+BUCKET+'/'+str(args['output_dir'])
    else:
      OUTPUTDIR = str(args['output_dir'])
    PROJECT = str(args['project'])
    JOBNAME = str(args['jobname'])#'alphatesting2'


    # images_blobs = get_blob(bucket_name, prefix)
    images_blobs = get_blob(BUCKET, IMAGE_FOLDER_PATH)
    with tf.gfile.Open('gs://testvideo_bucket_2/temp_folder/temp.csv', 'w') as csvfile:
        for blob in images_blobs:
            name = blob.name.split('/')[-1].split('.')[0]
            image_name = "gs://{}/{}/{}.jpg".format(BUCKET, IMAGE_FOLDER_PATH, name)
            label_name = "gs://{}/{}/{}.xml".format(BUCKET, LABEL_FOLDER_PATH, name)
            csvfile.write('{},{}\n'.format(image_name, label_name))


    staging_bucket = OUTPUTDIR+'/staging'

    temp_folder = OUTPUTDIR+'/temp'


    options = PipelineOptions()

    # # For Cloud execution, set the Cloud Platform project, job_name,
    # # staging location, temp_location and specify DataflowRunner.
    google_cloud_options = options.view_as(GoogleCloudOptions)
    worker_options = options.view_as(WorkerOptions)
    google_cloud_options.project = PROJECT
    google_cloud_options.job_name = JOBNAME
    google_cloud_options.staging_location = staging_bucket
    google_cloud_options.temp_location = temp_folder
    options.view_as(StandardOptions).runner = RUNNER
    worker_options.machine_type = 'n1-standard-16'
    worker_options.num_workers = 1
    # options.view_as(SetupOptions).save_main_session = True
   
    with beam.Pipeline(options=options) as p:
        _ = (
            p |
            'Read data from local' >> beam.io.ReadFromText('gs://testvideo_bucket_2/temp_folder/temp.csv') |
            # 'Convert data to tfrecord' >> beam.FlatMap(lambda line: convert_to_example(line)) |
            'Convert to tfrecord' >> beam.ParDo(ConvertTOExample())|
            'Save tfrecords' >> beam.io.tfrecordio.WriteToTFRecord(OUTPUTDIR+'/output_tfrecord.record', num_shards=1)
        )