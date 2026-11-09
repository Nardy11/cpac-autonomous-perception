#!/usr/bin/env python3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
import math
import cv2
import csv
import os
import numpy as np
import cvzone
from ultralytics import YOLO
import time
import pandas as pd
# from tracker import EuclideanDistTracker
# from kalmanfilter import KalmanFilter

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

class EuclideanDistTracker:
    def _init_(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 75:
                    self.center_points[id] = (cx, cy)
                   # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids




kf = KalmanFilter()

coff = np.polyfit([738, 676, 653, 569, 567], [40, 50, 60, 70, 80], 1)

# Create tracker object
tracker = EuclideanDistTracker()

subscriberName = 'Camera_detection'
topicName = 'video_topic'

model = YOLO("yolov8n.pt")
# model.export(format='onnx')
model.export(format='openvino')
# Load YOLOv8 model
ov_model = YOLO('yolov8n_openvino_model/')
video_filename = 'detection_video(V8)vino.avi'

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None


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

def callbackFunction(message):
    global out, frame_idx,trajectories,prev_gray

    bridgeObject = CvBridge()
    rospy.loginfo('received a video')
    convertedFrame = bridgeObject.imgmsg_to_cv2(message)

    convertedFrame=convertedFrame.copy()

    if out is None:
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(convertedFrame.shape[1]), int(convertedFrame.shape[0])))

    # Make detections
    results = ov_model(convertedFrame)
    detection = results[0].boxes.xyxy
    detectlist=[]

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

#--------------------------------------------------------------------------------------------------------------added part of optical flow  
    frame_gray = cv2.cvtColor(convertedFrame, cv2.COLOR_BGR2GRAY)
#--------------------------------------------------------------------------------------------------------------
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
                image_filename = f'frame_{frame_count}object{idx}.jpg'
                cv2.imwrite(image_filename, convertedFrame)
                image_filenames=image_filename

                # cv2.rectangle(convertedFrame, (int(top_left_x), int(bottom_right_y)), (int(bottom_right_x), int(top_left_y)), (0, 255, 0), 2)
                # cv2.putText(convertedFrame, f'Class: {label}', (int(top_left_x), int(bottom_right_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                #             2)
                width = int(bottom_right_x - top_left_x)
                height = int(top_left_y - bottom_right_y)
                detectlist.append([top_left_x, bottom_right_y, width, height])

                distance = int(math.sqrt((top_left_y - bottom_right_y) * 2 + (top_left_x - bottom_right_x) * 2))
                A, B = coff
                distance_cm = A * distance + B

                center = round(int(top_left_x + bottom_right_x) / 2), round(int(top_left_y + bottom_right_y) / 2)
                print ("center",center)
#--------------------------------------------------------------------------------------------------------------added part of prediction  

                predicted = kf.predict(center[0], center[1])
                cv2.circle(convertedFrame, predicted, 2, (255, 0, 0), 2)  # Draw predicted position

#--------------------------------------------------------------------------------------------------------------  

#--------------------------------------------------------------------------------------------------------------added part of optical flow  
                if len(trajectories) > 0:
                    img0, img1 = prev_gray, frame_gray
                    p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
                    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    # distance_moved = abs(p0-p0r).reshape(-1, 2)*(10/(frame_count / (time.time() - start_time)))
                    good = d < 1
    
                    # Calculate speed of keypoints
                    # speeds = np.linalg.norm(distance_moved, axis=1)*3.6

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
                        cv2.circle(convertedFrame, (int(x), int(y)), 2, (0, 0, 255), -1)
                        # cv2.putText(convertedFrame, f"{speed:.2f} km/h", (int(x) + 10, int(y) + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # print("speed:", speed)

                    trajectories = new_trajectories

                    # Draw all the trajectories
                    cv2.polylines(convertedFrame, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))

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
                print("Detected object:", label)
                print('Distance not correct:', distance)
                print('Real distance:', distance_cm)
                print("Top-Left corner:", f'( {top_left_x},{top_left_y} )')
                print("Top-Right corner:", f'( {bottom_right_x},{top_left_y} )')
                print("Bottom-Right corner:", f'( {bottom_right_x},{bottom_right_y} )')
                print("Bottom-Left corner:", f'( {top_left_x},{bottom_right_y} )')

                cv2.circle(convertedFrame, center, 2, (200, 200, 0), 2)
                # cv2.putText(convertedFrame, str(distance_cm) + " CM", (center[0] - 20, center[1] - 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                # cv2.putText(convertedFrame, 'Width: ' + str(width), (center[0] + 100, center[1] - 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                # cv2.putText(convertedFrame, 'Center', (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (200, 200, 0), 1)
                # cv2.putText(convertedFrame, 'Height: ' + str(height), (center[0] + 100, center[1] - 50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        #     cvzone.putTextRect(convertedFrame, str(distance_cm) + " CM", center, 1, 2, (200, 200, 0), 1)
       # 2. Object Tracking
    boxes_ids = tracker.update(detectlist)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(convertedFrame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(convertedFrame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    out.write(convertedFrame)
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
    cv2.imshow('YOLO_V8',cv2.resize(convertedFrame,(500,500)))

        #to put an ending sign for each iteration
    if cv2.waitKey(10) == ord('q'):
        out.release()
        cv2.destroyAllWindows()
        

rospy.init_node(subscriberName, anonymous=True)
rospy.Subscriber(topicName, Image, callbackFunction)
rospy.spin()