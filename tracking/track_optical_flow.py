import math
import cv2
import csv
import os
import numpy as np
import cvzone
from ultralytics import YOLO
import time
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', default='0', help='camera index or video path')
args = parser.parse_args()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
model.export(format='openvino')
# Coefficients for distance calculation
from tracker import *
# from realsense_depth import *  # unused; module not part of the original upload
# Create tracker object
tracker = EuclideanDistTracker()
coff = np.polyfit([738, 676, 653, 569, 567], [40, 50, 60, 70, 80], 1)
ov_model=YOLO('yolov8n_openvino_model')
filename = 'yolov8_outputs1vino.csv'
video_filename = 'detection_video(V8)vino.avi'
width=500
height=500
# Check if file exists
file_exists = os.path.isfile(filename)

#--------------------------------------------------------------------------------------------------------------added part of optical flow  

lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.7,
                    minDistance = 10,
                    blockSize = 7 )


trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0


#-------------------------------------------------------------------------------------------------------------- 



# Open CSV file for appending or create a new file
with open(filename, mode='a', newline='') as csvfile:
    fieldnames = ['Name Of Objects', 'Top Left Borders', 'Top Right Borders', 'Bottom Left Borders',
                  'Bottom Right Borders', 'Height', 'Width', 'Center', 'Distance(cm)','Frame']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # If file is new, write the header
    if not file_exists:
        writer.writeheader()
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    # cap.set(cv2.CAP_PROP_EXPOSURE, -100) #to adjust HD3000 Microsoft USB camera
    cap.set(3, 180)
    cap.set(4, 180)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0,(int(cap.get(3)), int(cap.get(4))))
    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1

#--------------------------------------------------------------------------------------------------------------added part of optical flow  
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#--------------------------------------------------------------------------------------------------------------
        
        # Make detections
        results = ov_model(frame)
        detection = results[0].boxes.xyxy
        detectlist=[]
        if len(detection) > 0:
            for idx,detect in enumerate(detection):
                if int(results[0].boxes.data[idx][5])==0:
                    top_left_x = int(detect[0])
                    top_left_y = int(detect[3])
                    bottom_right_x = int(detect[2])
                    bottom_right_y = int(detect[1])
                    label_index = int(results[0].boxes.data[idx][5])
                    label = results[0].names[int(label_index)]

                    # Save frame as image
                    image_filename = f'frame_{frame_count}_object_{idx}.jpg'
                    cv2.imwrite(image_filename, frame)
                    image_filenames=image_filename

                    # cv2.rectangle(frame, (int(top_left_x), int(bottom_right_y)), (int(bottom_right_x), int(top_left_y)), (0, 255, 0), 2)
                    # cv2.putText(frame, f'Class: {label}', (int(top_left_x), int(bottom_right_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    #             2)
                    width = int(bottom_right_x - top_left_x)
                    height = int(top_left_y - bottom_right_y)
                    detectlist.append([top_left_x, bottom_right_y, width, height])

                    distance = int(math.sqrt((top_left_y - bottom_right_y) ** 2 + (top_left_x - bottom_right_x) ** 2))
                    A, B = coff
                    distance_cm = A * distance + B

                    center = round(int(top_left_x + bottom_right_x) / 2), round(int(top_left_y + bottom_right_y) / 2)
                    print ("center",center)
#--------------------------------------------------------------------------------------------------------------added part of optical flow  
                    if len(trajectories) > 0:
                        img0, img1 = prev_gray, frame_gray
                        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
                        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                        d = abs(p0-p0r).reshape(-1, 2).max(-1)
                        good = d < 1

                        new_trajectories = []

                        # Get all the trajectories
                        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
                            if not good_flag:
                                continue
                            trajectory.append((x, y))
                            if len(trajectory) > trajectory_len:
                                del trajectory[0]
                            new_trajectories.append(trajectory)
                            # Newest detected point
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

                        trajectories = new_trajectories

                        # Draw all the trajectories
                        cv2.polylines(frame, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
                        cv2.putText(frame, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

                    # Update interval - When to update and detect new features
                    if frame_idx % detect_interval == 0:
                        mask = np.zeros_like(frame_gray)
                        mask[:] = 255

                        # Lastest point in latest trajectory
                        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
                            cv2.circle(mask, (x, y), 5, 0, -1)

                        # Detect the good features to track
                        p = np.array([[center[0],center[1]]],dtype=np.float32)
                        print('p =',  p)
                        if p is not None:
                            # If good features can be tracked - add that to the trajectories
                            for x, y in np.float32(p).reshape(-1, 2):
                                trajectories.append([(x, y)])
#--------------------------------------------------------------------------------------------------------------


                    writer.writerow({'Name Of Objects': label,
                                    'Top Left Borders': f'({top_left_x},{top_left_y})',
                                    'Top Right Borders': f'({bottom_right_x},{top_left_y})',
                                    'Bottom Left Borders': f'({top_left_x},{bottom_right_y})',
                                    'Bottom Right Borders': f'({bottom_right_x},{bottom_right_y})',
                                    'Height': height,
                                    'Width': width,
                                    'Center': center,
                                    'Distance(cm)': distance_cm,
                                    'Frame': image_filenames})

                    print("Detected object:", label)
                    print('Distance not correct:', distance)
                    print('Real distance:', distance_cm)
                    print("Top-Left corner:", f'( {top_left_x},{top_left_y} )')
                    print("Top-Right corner:", f'( {bottom_right_x},{top_left_y} )')
                    print("Bottom-Right corner:", f'( {bottom_right_x},{bottom_right_y} )')
                    print("Bottom-Left corner:", f'( {top_left_x},{bottom_right_y} )')

                    cv2.circle(frame, center, 5, (200, 200, 0), 5)
                    cv2.putText(frame, str(distance_cm) + " CM", (center[0] - 20, center[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    cv2.putText(frame, 'Width: ' + str(width), (center[0] + 100, center[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    cv2.putText(frame, 'Center', (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 0), 1)
                    cv2.putText(frame, 'Height: ' + str(height), (center[0] + 100, center[1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            #     cvzone.putTextRect(frame, str(distance_cm) + " CM", center, 1, 2, (200, 200, 0), 1)
           # 2. Object Tracking
        boxes_ids = tracker.update(detectlist)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        out.write(frame)
        # Once the loop ends, write the list of image filenames to a CSV file
        
#--------------------------------------------------------------------------------------------------------------added part of optical flow  
        frame_idx += 1
        prev_gray = frame_gray
# -------------------------------------------------------------------------------------------------------------
        # Calculate and print FPS
        if time.time() - start_time >= 1:
            fps = frame_count / (time.time() - start_time)
            print(f'FPS: {fps:.2f}')
            # Reset variables for next second
            start_time = time.time()
            frame_count = 0
        cv2.imshow('YOLO_V8',cv2.resize(frame,(500,500)))

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

    out.release()
    cap.release()
    cv2.destroyAllWindows()