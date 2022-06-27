#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:23:20 2019

@author:Abrar Istiak Akib
"""

import tensorflow as tf
import numpy as np
import os
import sys
import tarfile
import cv2
import time
import argparse

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inp",type=str,default="test-directory-folder",
help="path to input video")
ap.add_argument("-o", "--out", type=str, default='protected-directory-folder',
help="Output folder")

args = vars(ap.parse_args())

# This is needed since the notebook is stored in the submissionPackage folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util

MODEL_NAME = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def blurring(image):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    start=time.time()
    im_pil=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im_pil)
    image_np = load_image_into_numpy_array(im_pil)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    (H, W) = im_pil.size
    class_idd=output_dict['detection_classes']
    maskRI=output_dict.get('detection_masks')
    maskRI=maskRI>0
    x,y,z=maskRI.shape
    dummyat=np.zeros((W,H),dtype=bool)
    class_idd=class_idd[0:x]

    for ix in range(x):
        if(class_idd[ix]==1 or class_idd[ix]==72 or class_idd[ix]==73 or class_idd[ix]==70 or class_idd[ix]==76 or class_idd[ix]==77 or class_idd[ix]==84):
            dummyat=dummyat|maskRI[ix,:,:]
        else:
            continue
    img_copy=image
    blur_img=cv2.GaussianBlur(img_copy,(45,45),0)
    img_copy[dummyat]=blur_img[dummyat]
    end=time.time()
    print("Took {} seconds to process".format(end-start))
    return img_copy


def video_from_dir(dir):
    temp=os.listdir(dir)   #give input directory
    video=[]
    for i in temp:
        if(i.endswith('.MP4')):
            video.append(i)
    return video


if __name__ == '__main__':
    cur_dir=os.getcwd()
    input_path=os.path.join(cur_dir,args["inp"])         #whatever test_directory is named
    video=video_from_dir(input_path)
    for xy in range(len(video)):
        print(video[xy])
        class_id=[]
        #name.append(video[xy])
        capture = cv2.VideoCapture(os.path.join(input_path,video[xy]))
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        codec = cv2.VideoWriter_fourcc(*'MP4V')
        os.chdir(os.path.join(cur_dir, args["out"]))          #whatever protected saved data folder name is 
        output = cv2.VideoWriter(video[xy], codec, fps, size)
        ret, frame = capture.read()
        while(ret):
            img_processed = blurring(frame)
            output.write(img_processed)
            #cv2.imshow('frame',img_processed)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            ret, frame = capture.read()
        capture.release()
        output.release()
        cv2.destroyAllWindows()
