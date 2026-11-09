import os
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
import tensorflow as tf
import time
import math
import csv
from object_detection.utils import visualization_utils as viz_utils

from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tfod-root', required=True, help='root of the TF Object Detection workspace (contains workspace/, scripts/, models/)')
args = parser.parse_args()

TFOD_ROOT = args.tfod_root
#pathes
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH':os.path.join(TFOD_ROOT, 'workspace'),
    'SCRIPTS_PATH': os.path.join(TFOD_ROOT, 'scripts'),
    'APIMODEL_PATH': os.path.join(TFOD_ROOT, 'models'),
    'ANNOTATION_PATH': os.path.join(TFOD_ROOT, 'workspace/annotations'),
    'IMAGE_PATH': os.path.join(TFOD_ROOT, 'workspace/images'),
    'MODEL_PATH': os.path.join(TFOD_ROOT, 'workspace/models'),
    'PRETRAINED_MODEL_PATH':os.path.join(TFOD_ROOT, 'workspace/pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(TFOD_ROOT, 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join(TFOD_ROOT, 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join(TFOD_ROOT, 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join(TFOD_ROOT, 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join(TFOD_ROOT,'protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-51')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load your object detection model (you've already done this)
# Replace 'detect_fn' with your actual detection function
# Example: detect_fn = load_your_model()

# Coefficients for distance calculation
coff = np.polyfit([738, 676, 653, 569, 567], [40, 50, 60, 70, 80], 1)

filename = 'Tensorflow2_outputs.csv'
video_filename = 'detection_video(TF2).avi'

# Check if file exists
file_exists = os.path.isfile(filename)
# Open CSV file for appending or create a new file
with open(filename, mode='a', newline='') as csvfile:
    fieldnames = ['Name Of Objects', 'Top Left Borders', 'Top Right Borders', 'Bottom Left Borders',
                  'Bottom Right Borders', 'Height', 'Width', 'Center', 'Distance(cm)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
    # If file is new, write the header
    if not file_exists:
        writer.writeheader()

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        detection_boxes=detections['detection_boxes']

        # Visualize the detected objects
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detection_boxes,
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=.8,
            agnostic_mode=False)

        if np.any(detections['detection_scores'] > 0.8):
            centers = []
            for i, box in enumerate(detection_boxes):
                ymin, xmin, ymax, xmax = box
                confidence = detections['detection_scores'][i]
                if confidence > 0.8:
                    left = int(xmin * width)
                    bottom = int(ymin * height)
                    right = int(xmax * width)
                    top = int(ymax * height)
                    class_id = int(detections['detection_classes'][i]+ label_id_offset)
                    class_name = category_index[class_id]['name']
                    center_x = int((xmin + xmax) / 2 * width)
                    center_y = int((ymin + ymax) / 2 * height)
                    centers.append((center_x, center_y))
                    cv2.circle(image_np_with_detections, (center_x, center_y), 5, (200, 200, 0), 5)
                    distance = int(math.sqrt((top - bottom) ** 2 + (right - left) ** 2))
                    A, B = coff
                    distance_cm = A * distance + B
                    cv2.putText(image_np_with_detections, str(distance_cm) + " CM", (center_x - 20, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    cv2.putText(image_np_with_detections, 'Width: ' + str(int(right-left)), (center_x + 100, center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    cv2.putText(image_np_with_detections, 'Center', (center_x - 20, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 0), 1)
                    cv2.putText(image_np_with_detections, 'Height: ' + str(int(top-bottom)), (center_x + 100, center_y - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

                    writer.writerow({'Name Of Objects': class_name,
                                     'Top Left Borders': f'({left},{top})',
                                     'Top Right Borders': f'({right},{top})',
                                     'Bottom Left Borders': f'({left},{bottom})',
                                     'Bottom Right Borders': f'({right},{bottom})',
                                     'Height': int(top-bottom),
                                     'Width': int(right-left),
                                     'Center': (center_x,center_y),
                                     'Distance(cm)': distance_cm})
        out.write(image_np_with_detections)
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            writer.writerow({'Name Of Objects': '-----------',
                             'Top Left Borders': '-----------',
                             'Top Right Borders': '-----------',
                             'Bottom Left Borders': '-----------',
                             'Bottom Right Borders': '-----------',
                             'Height': '-----------',
                             'Width': '-----------',
                             'Center': '-----------',
                             'Distance(cm)': '-----------'})
            break

        # Check if one second has passed
        if time.time() - start_time >= 1:
            fps = frame_count / (time.time() - start_time)
            print(f'FPS: {fps:.2f}')
            # Reset variables for the next second
            start_time = time.time()
            frame_count = 0

    cap.release()
    cv2.destroyAllWindows()
