import cvzone
import numpy as np
import argparse
import tensorflow as tf
import cv2
import time
import math
import os
import csv
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
width = 500
height = 500
# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

coff = np.polyfit([738, 676, 653, 569, 567], [40, 50, 60, 70, 80], 1)

def run_inference(model, category_index, cap):
    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image = np.array(frame)
        image = image.astype(np.uint8)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        # Run inference
        # Actual detection.
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)

        # Run inference using the model signature
        output_dict = model(tf.constant(image_batch))
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # label_id_offset = 1
        image_np_with_detections = cv2.resize(image.copy(),(width, height))
        detection_boxes = output_dict['detection_boxes']
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=2)

        if np.any(output_dict['detection_scores'] > 0):
            for i, box in enumerate(detection_boxes):
                ymin, xmin, ymax, xmax = box
                confidence = output_dict['detection_scores'][i]
                if confidence > 0.3:
                    left = int(xmin * width)
                    bottom = int(ymin * height)
                    right = int(xmax * width)
                    top = int(ymax * height)

                    # Calculate center coordinates
                    center_x = (left + right) // 2
                    center_y = (top + bottom) // 2

                    # Decrease the size of the text
                    font_scale = 0.5  # Change the font scale to 0.5
                    image_filename = f'frame_{frame_count}_object_{i}.jpg'
                    cv2.imwrite(image_filename, image_np_with_detections)
                    image_filenames = image_filename
                    # Draw rectangle and text
                    cv2.rectangle(image_np_with_detections, (left, top), (right, bottom), (200, 200, 0), 2)
                    cv2.circle(image_np_with_detections, (center_x, center_y), 5, (200, 200, 0), 2)
                    cv2.putText(image_np_with_detections, 'Width: ' + str(right - left), (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 0), 1)
                    cv2.putText(image_np_with_detections, 'Center', (center_x - 20, center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 0), 1)
                    cv2.putText(image_np_with_detections, 'Height: ' + str(top - bottom), (left, top - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 0), 1)

                    # Add class name on top of the bounding box with color (200, 200, 0)
                    class_name = category_index[output_dict['detection_classes'][i]]['name']
                    # cv2.putText(image_np_with_detections, class_name, (left, top),
                                # cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 0), 1)
                    cvzone.putTextRect(image_np_with_detections, f'Class: {class_name}    {confidence * 100:.2f}%', (left, bottom), 1, 2, (200, 200, 0), 1)
                    distance = int(math.sqrt((top - bottom) ** 2 + (right - left ) ** 2))
                    A, B = coff
                    distance_cm = A * distance + B
                    writer.writerow({'Name Of Objects': class_name,
                                     'Top Left Borders': f'({left},{top})',
                                     'Top Right Borders': f'({right},{top})',
                                     'Bottom Left Borders': f'({left},{bottom})',
                                     'Bottom Right Borders': f'({right},{bottom})',
                                     'Height': bottom - top,
                                     'Width': right - left,
                                     'Center': (center_x, center_y),
                                     'Distance(cm)': distance_cm,
                                     'Frame': image_filenames})

        out.write(image_np_with_detections)
        if time.time() - start_time >= 1:
            fps = frame_count / (time.time() - start_time)
            print(f'FPS: {fps:.2f}')
            # Reset variables for the next second
            start_time = time.time()
            frame_count = 0
        cv2.imshow('object detection',image_np_with_detections )

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-v', '--video_path', type=str, required=True, help='Path to video.')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    cap = cv2.VideoCapture(args.video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Set frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Set frame height
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate

    filename = 'Tensorflow2_outputs.csv'
    video_filename = 'detection_video(TF2).avi'

    # Check if file exists
    file_exists = os.path.isfile(filename)
    # Open CSV file for appending or create a new file
    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = ['Name Of Objects', 'Top Left Borders', 'Top Right Borders', 'Bottom Left Borders',
                      'Bottom Right Borders', 'Height', 'Width', 'Center', 'Distance(cm)', 'Frame']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))  # Set output dimensions to match input video
        # If file is new, write the header
        if not file_exists:
            writer.writeheader()

        try:
            run_inference(detection_model, category_index, cap)
        except KeyboardInterrupt:
            print("Process terminated by user.")
        finally:
            # Release resources
            out.release()
            cap.release()
            cv2.destroyAllWindows()

# python .\detect_from_video.py -m ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model -l .\data\mscoco_label_map.pbtxt -v .\myVideo.mp4