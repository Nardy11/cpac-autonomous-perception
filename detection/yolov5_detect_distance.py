import math
import cvzone
import torch
import numpy as np
import cv2
import csv
import os
import time
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--weights', default='weights/yolov5s_custom_last.pt', help='YOLOv5 custom weights')
parser.add_argument('--source', default='0', help='camera index or video path')
args = parser.parse_args()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', args.weights)

# Coefficients for distance calculation
coff = np.polyfit([738, 676, 653, 569, 567], [40, 50, 60, 70, 80], 1)

filename = 'yolov5_outputs.csv'
video_filename = 'detection_video(V5).avi'

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

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    # cap.set(cv2.CAP_PROP_EXPOSURE, -9.5555555)#to adjust HD3000 Microsoft USB camera

    cap.set(3, 640)
    cap.set(4, 480)
    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1

        # Make detections
        results = model(frame)
        detection = results.xyxy[0]
        if len(detection) > 0:
            top_left_x = int(detection[0][0])
            top_left_y = int(detection[0][3])
            bottom_right_x = int(detection[0][2])
            bottom_right_y = int(detection[0][1])
            width = int(bottom_right_x - top_left_x)
            height = int(top_left_y - bottom_right_y)
            label_index = int(detection[0][5])
            label = results.names[label_index]
            distance = int(math.sqrt((top_left_y - bottom_right_y) ** 2 + (top_left_x - bottom_right_x) ** 2))
            A, B = coff
            distance_cm = A * distance + B

            center = round(int(top_left_x + bottom_right_x) / 2), round(int(top_left_y + bottom_right_y) / 2)
            cv2.circle(frame, center, 5, (200, 200, 0), 5)
            writer.writerow({'Name Of Objects': label,
                             'Top Left Borders': f'({top_left_x},{top_left_y})',
                             'Top Right Borders': f'({bottom_right_x},{top_left_y})',
                             'Bottom Left Borders': f'({top_left_x},{bottom_right_y})',
                             'Bottom Right Borders': f'({bottom_right_x},{bottom_right_y})',
                             'Height': height,
                             'Width': width,
                             'Center': center,
                             'Distance(cm)': distance_cm})

            print("Detected object:", label)
            print('Distance not correct:', distance)
            print('Real distance:', distance_cm)
            print("Top-Left corner:", f'( {top_left_x},{top_left_y} )')
            print("Top-Right corner:", f'( {bottom_right_x},{top_left_y} )')
            print("Bottom-Right corner:", f'( {bottom_right_x},{bottom_right_y} )')
            print("Bottom-Left corner:", f'( {top_left_x},{bottom_right_y} )')

            cv2.putText(frame, str(distance_cm) + " CM", (center[0] - 20, center[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv2.putText(frame, 'Width: ' + str(width), (center[0] + 100, center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv2.putText(frame, 'Center', (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 0), 1)
            cv2.putText(frame, 'Height: ' + str(height), (center[0] + 100, center[1] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        #     cvzone.putTextRect(frame, str(distance_cm) + " CM", center, 1, 2, (200, 200, 0), 1)
        out.write(frame)
        cv2.imshow('YOLO_V5', np.squeeze(results.render()))
        # Calculate and print FPS
        if time.time() - start_time >= 1:
            fps = frame_count / (time.time() - start_time)
            print(f'FPS: {fps:.2f}')
            # Reset variables for next second
            start_time = time.time()
            frame_count = 0

        #to put an ending sign for each iteration
        if cv2.waitKey(10) == ord('q'):
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
    cap.release()
    out.release()
    cv2.destroyAllWindows()