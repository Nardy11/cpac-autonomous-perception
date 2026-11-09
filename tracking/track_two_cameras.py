import math
import cv2
import csv
import os
import numpy as np
# import cvzone
from ultralytics import YOLO
import time
import triangulation  # stereo depth helper (find_depth); not part of the original upload
# import pandas as pd
# Load YOLOv8 model
# model.fuse()
# model.export(format='openvino')
# Coefficients for distance calculation
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()
coff = np.polyfit([738, 676, 653, 569, 567], [40, 50, 60, 70, 80], 1)
ov_model=YOLO('yolov8n.pt')
# ov_model.fuse()
filename = 'yolov8_outputs1.csv'
video_filename = 'detection_video(V8).avi'
width=300
height=300

centerF_x = width // 2
centerF_y = height // 2

def get_angle(center_x, center_y, point_x, point_y):
    # Calculate the displacement of the point relative to the center
    delta_x = point_x - center_x
    delta_y = point_y - center_y

    # Calculate the angle using arctan2
    angle_radians = math.atan2(delta_y, delta_x)

    # Convert the angle from radians to degrees
    angle_degrees = math.degrees(angle_radians)

    # Ensure the angle is in the range [0, 360)
    angle_degrees = angle_degrees % 90

    return angle_degrees
# Check if file exists
file_exists = os.path.isfile(filename)
# Open CSV file for appending or create a new file

with open(filename, mode='a', newline='') as csvfile:
    fieldnames = ['Name Of Objects', 'Top Left Borders', 'Top Right Borders', 'Bottom Left Borders',
                  'Bottom Right Borders', 'Height', 'Width', 'Center', 'Distance(cm)','Frame']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # If file is new, write the header
    if not file_exists:
        writer.writeheader()
    cap = cv2.VideoCapture(1)
    cap1 = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_EXPOSURE, -100) #to adjust HD3000 Microsoft USB camera
    cap.set(3, 180)
    cap.set(4, 180)
    cap1.set(3, 180)

    cap1.set(4, 180)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0,(int(cap.get(3)), int(cap.get(4))))
    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()
        detectlist,detectlist1=[],[]
        frame_count += 1
        f1=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f1=cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
        f2=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        f2=cv2.cvtColor(f2, cv2.COLOR_GRAY2BGR)
        # Make detections
        results = ov_model(f1)
        results1 = ov_model(f2)
        detection = results[0].boxes.xyxy
        detection1 = results1[0].boxes.xyxy                
            #     cvzone.putTextRect(frame, str(distance_cm) + " CM", center, 1, 2, (200, 200, 0), 1)
        # Once the loop ends, write the list of image filenames to a CSV file
        if len(detection) > 0:
            for idx, (detect, detect1) in enumerate(zip(detection, detection1)):
                if int(results1[0].boxes.data[idx][5])==0:
                    top_left_x1 = int(detect1[0])
                    top_left_y1 = int(detect1[3])
                    bottom_right_x1 = int(detect1[2])
                    bottom_right_y1 = int(detect1[1])
                    label_index1 = int(results1[0].boxes.data[idx][5])
                    label1 = results1[0].names[int(label_index1)]
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                    cv2.rectangle(f2, (int(top_left_x1), int(bottom_right_y1)), (int(bottom_right_x1), int(top_left_y1)), (0, 255, 0), 2)
                    cv2.putText(f2, f'Class: {label1}', (int(top_left_x1), int(bottom_right_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                    width1 = int(bottom_right_x1 - top_left_x1)
                    height1 = int(top_left_y1 - bottom_right_y1)
                    distance1 = int(math.sqrt((top_left_y1 - bottom_right_y1) ** 2 + (top_left_x1 - bottom_right_x1) ** 2))
                    A1, B1 = coff
                    distance_cm1 = A1 * distance1 + B1
                    center1 = round(int(top_left_x1 + bottom_right_x1) / 2), round(int(top_left_y1 + bottom_right_y1) / 2)
                    angle1=get_angle(centerF_x,centerF_y,center1[0],center1[1])
                    detectlist1.append([top_left_x1, bottom_right_y1, width1, height1])
                #------------------------------------------------------------------------------------------------------------------------------------------------ 
                    #Object Tracking of frame 1
                    boxes_ids1 = tracker.update(detectlist1)
                    for box_id in boxes_ids1:
                        x, y, w, h, id = box_id
                        print(str(id))
                        cv2.putText(frame1, str(id), (center1[0] - 20,  center1[1] - 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                #------------------------------------------------------------------------------------------------------------------------------------------------
                    # print('angel1:',angle1)
                    print("Detected object1:", label1)
                    # print('Distance not correct:', distance1)
                    # print('Real distance1:', distance_cm1)
                    # print("Top-Left corner:1", f'( {top_left_x1},{top_left_y1} )')
                    # print("Top-Right corner1:", f'( {bottom_right_x1},{top_left_y1} )')
                    # print("Bottom-Right corner1:", f'( {bottom_right_x1},{bottom_right_y1} )')
                    # print("Bottom-Left corner:1", f'( {top_left_x1},{bottom_right_y1} )')
                    # print()
                    cv2.circle(f2, center1, 5, (200, 200, 0), 5)
                    # cv2.putText(frame1, str(distance_cm1) + " CM", (center1[0] - 20, center1[1] - 40),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    # cv2.putText(frame1, 'Width:1 ' + str(width1), (center1[0] + 100, center1[1] - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    # cv2.putText(frame1, 'Center angle1:'+str(angle1), (center1[0] - 20, center1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.5, (200, 200, 0), 1)
                    # cv2.putText(frame1, 'Height1: ' + str(height1), (center1[0] + 100, center1[1] - 50),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                else:
                    center1 = 0,0
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                if int(results [0].boxes.data[idx][5])==0:
                    top_left_x = int(detect[0])
                    top_left_y = int(detect[3])
                    bottom_right_x = int(detect[2])
                    bottom_right_y = int(detect[1])
                    label_index = int(results[0].boxes.data[idx][5])
                    label = results[0].names[int(label_index)]
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                    cv2.rectangle(f1, (int(top_left_x), int(bottom_right_y)), (int(bottom_right_x), int(top_left_y)), (0, 255, 0), 2)
                    cv2.putText(f1, f'Class: {label}', (int(top_left_x), int(bottom_right_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                    width = int(bottom_right_x - top_left_x)
                    height = int(top_left_y - bottom_right_y)
                    distance = int(math.sqrt((top_left_y - bottom_right_y) ** 2 + (top_left_x - bottom_right_x) ** 2))
                    A, B = coff
                    distance_cm = A * distance + B
                    center = round(int(top_left_x + bottom_right_x) / 2), round(int(top_left_y + bottom_right_y) / 2)
                    angle=get_angle(centerF_x,centerF_y,center[0],center[1])
                    detectlist.append([top_left_x, bottom_right_y, width, height])
                #------------------------------------------------------------------------------------------------------------------------------------------------
                    #Object Tracking of frame 0
                    boxes_ids = tracker.update(detectlist)
                    for box_id in boxes_ids:
                        x, y, w, h, id = box_id
                        print(str(id))
                        cv2.putText(frame, str(id), (center[0] - 20,  center[1] - 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                #------------------------------------------------------------------------------------------------------------------------------------------------
                    # # Save frame as image
                    # image_filename = f'frame_{frame_count}_object_{idx}.jpg'
                    # cv2.imwrite(image_filename, f1)
                    # image_filenames=image_filename
                    writer.writerow({'Name Of Objects': label,
                                    'Top Left Borders': f'({top_left_x},{top_left_y})',
                                    'Top Right Borders': f'({bottom_right_x},{top_left_y})',
                                    'Bottom Left Borders': f'({top_left_x},{bottom_right_y})',
                                    'Bottom Right Borders': f'({bottom_right_x},{bottom_right_y})',
                                    'Height': height,
                                    'Width': width,
                                    'Center': center,
                                    'Distance(cm)': distance_cm,
                                    'Frame': ''})
                    # ------------------------------------------------------------------------------------------------------------------------------------------------                
                    # print('angel:',angle)
                    print("Detected object:", label)
                    # print('Distance not correct:', distance)
                    # print('Real distance:', distance_cm)
                    # print("Top-Left corner:", f'( {top_left_x},{top_left_y} )')
                    # print("Top-Right corner:", f'( {bottom_right_x},{top_left_y} )')
                    # print("Bottom-Right corner:", f'( {bottom_right_x},{bottom_right_y} )')
                    # print("Bottom-Left corner:", f'( {top_left_x},{bottom_right_y} )')
                    # print()
                    cv2.circle(f1, center, 5, (200, 200, 0), 5)

                    # cv2.putText(frame, str(distance_cm) + " CM", (center[0] - 20, center[1] - 40),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    # cv2.putText(frame, 'Width: ' + str(width), (center[0] + 100, center[1] - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    # cv2.putText(frame, 'Center angle:'+str(angle), (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.5, (200, 200, 0), 1)
                    # cv2.putText(frame, 'Height: ' + str(height), (center[0] + 100, center[1] - 50),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    # ------------------------------------------------------------------------------------------------------------------------------------------------
                else:
                    center=0,0    
                distanceeee =triangulation.find_depth(center,center1,f1,f2) 
                print("distance:",distanceeee)    
                # cv2.putText(f1, str(distanceeee) + " CM", (center[0] - 20, center[1] - 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                # cv2.putText(f2, str(distanceeee) + " CM", (center1[0] - 20, center1[1] - 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            #Object Tracking of frame 1
        boxes_ids1 = tracker.update(detectlist1)
        for box_id in boxes_ids1:
            x, y, w, h, id = box_id
            cv2.putText(f2, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(f2, (x, y), (x + w, y + h), (0, 255, 0), 3)                
                #------------------------------------------------------------------------------------------------------------------------------------------------
            #Object Tracking of frame 0
            boxes_ids = tracker.update(detectlist)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(f1, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(f1, (x, y), (x + w, y + h), (0, 255, 0), 3)                
                #------------------------------------------------------------------------------------------------------------------------------------------------

        out.write(f1)
        # # Calculate and print FPS
        # if time.time() - start_time >= 1:
        #     fps = frame_count / (time.time() - start_time)
        #     print(f'FPS: {fps:.2f}')
        #     # Reset variables for next second
        #     start_time = time.time()
        #     frame_count = 0
        cv2.imshow('YOLO_V8_left',cv2.resize(f1,(300,300)))
        cv2.imshow('YOLO_V8_right',cv2.resize(f2,(300,300)))

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