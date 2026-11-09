import math
import cv2
import csv
import os
import numpy as np
from ultralytics import YOLO
import time
import pyzed.sl as sl
import pandas as pd
# Load YOLOv8 model

# Coefficients for distance calculation
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))
from tracker import *
# from realsense_depth import *  # unused; module not part of the original upload
# Create tracker object


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA

    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.CENTIMETER  # Use meter units (for depth measurements)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened successfully
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()
    model = YOLO("yolov8n.pt")
    model.export(format='openvino')
    tracker = EuclideanDistTracker()
    ov_model=YOLO('yolov8n_openvino_model')
    width=500
    height=500
    start_time = time.time()
    frame_count = 0

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

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()

    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))

    # Create a window to display the left camera view

    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            frame_count+=1
            # Retrieve left image
            zed.retrieve_image(image,sl.VIEW.RIGHT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            image_ocv = image.get_data()
            # Convert the image to RGB format if it's in BGR format
            frame = cv2.cvtColor(cv2.cvtColor(image_ocv, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)

            # Resize the image to match the expected input size of the model (640x640)
            frame_resized = cv2.resize(frame, (500, 500))
            frame = frame_resized
            # Pass the resized image to the YOLOv8 model
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = frame_gray
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

                        width = int(bottom_right_x - top_left_x)
                        height = int(top_left_y - bottom_right_y)
                        detectlist.append([top_left_x, bottom_right_y, width, height])
                        center = round(int(top_left_x + bottom_right_x) / 2), round(int(top_left_y + bottom_right_y) / 2)
                        # print ("center",center)
                        # Get and print distance value in mm at the center of the image
                        # We measure the distance camera - object using Euclidean distance
                        x = center[0]
                        y = center[1]
                        err, point_cloud_value = point_cloud.get_value(x, y)
                        if math.isfinite(point_cloud_value[2]):
                            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                                point_cloud_value[1] * point_cloud_value[1] +
                                                point_cloud_value[2] * point_cloud_value[2])
                            print(f"Distance to Camera at {{{x};{y}}}: {distance}")
                        else:
                            distance =0
                            print(f"The distance can not be computed at {{{x};{y}}}")
                        print('----------------------------------------------------')
                        angle= (center[0]/250)*90
                        print('----------------------------------------------------')
                        print('Angle',angle)
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
                                # cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

                            trajectories = new_trajectories

                            # Draw all the trajectories
                            cv2.polylines(frame, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
                            # cv2.putText(frame, 'dostance: ' + str(dostance)+' CM', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

                        # Update interval - When to update and detect new features
                        if frame_idx % detect_interval == 0:
                            mask = np.zeros_like(frame_gray)
                            mask[:] = 255

                            # Lastest point in latest trajectory
                            # for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
                            #     cv2.circle(mask, (x, y), 5, 0, -1)

                            # Detect the good features to track
                            p = np.array([[center[0],center[1]]],dtype=np.float32)
                            # print('p =',  p)
                            if p is not None:
                                # If good features can be tracked - add that to the trajectories
                                for x, y in np.float32(p).reshape(-1, 2):
                                    trajectories.append([(x, y)])
    #--------------------------------------------------------------------------------------------------------------
                        print("Detected object:", label)
                        cv2.circle(frame, center, 5, (200, 200, 0), 5)
                        cv2.putText(frame, 'distance: ' + str(distance), (center[0], center[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, 'angle: ' + str(angle), (center[0], center[1] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255),2)
                        # cv2.putText(frame, 'Width: ' + str(width), (center[0] + 100, center[1] - 20),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                        # cv2.putText(frame, 'Center', (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        #             0.5, (200, 200, 0), 1)
                        # cv2.putText(frame, 'Height: ' + str(height), (center[0] + 100, center[1] - 50),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)                        
            boxes_ids = tracker.update(detectlist)
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
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

        #to put an ending sign for each iteration

            cv2.imshow('Left Camera View', frame)


            # Convert the ZED image to OpenCV format

            # Display the left camera view

            # Wait for a key press and check if 'q' is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the OpenCV windows and the ZED camera
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
