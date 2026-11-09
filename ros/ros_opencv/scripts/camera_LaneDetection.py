#!/usr/bin/env python3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import tensorflow as tf
import keras
import os
from std_msgs.msg import Float32MultiArray

model =keras.models.load_model(os.environ.get('LANE_MODEL', 'model.h5'))

topicName = 'color_video_topic'
topicName_depth = 'depth_video_topic'
deptharray_topicName = 'depth_array_topic'

# Create windows for color and depth frames
cv2.namedWindow('color', cv2.WINDOW_NORMAL)
cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)

color_frame = None
depth_frame = None
depth_array = None
frame =None
video =None
result =None

def calculate_angle(object_center, frame_center, fov_horizontal_deg, fov_vertical_deg, frame_width, frame_height):
    # Calculate horizontal angle
    horizontal_pixel_offset = object_center[0] - frame_center[0]
    horizontal_angle = (horizontal_pixel_offset / frame_width) * fov_horizontal_deg

    # Calculate vertical angle
    vertical_pixel_offset = object_center[1] - frame_center[1]
    vertical_angle = (vertical_pixel_offset / frame_height) * fov_vertical_deg

    return horizontal_angle, vertical_angle


class lanes():
  def __init__(self):
    self.recent_fit =[]
    self.avg_fit =[]

def road_lines():
    global frame,depth_array
    # Resize the image to the expected shape
    if frame is not None:

        smallimg = cv2.resize(frame, (160, 80))  # Corrected resizing dimensions
        smallimg = np.array(smallimg)
        smallimg = smallimg[None,:,:,:]

        # Assuming 'model' is defined and loaded somewhere in your code
        prediction = model.predict(smallimg)[0] * 255
        print(prediction)
        # print(prediction)
        lanes.recent_fit.append(prediction)

        if len(lanes.recent_fit) > 5:
            lanes.recent_fit = lanes.recent_fit[1:]

        lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)
        blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
        lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
        # print('lines',lane_drawn)
        # Resize the lane image to match the size of the original image
        lane_image = cv2.resize(lane_drawn, (500,500))
        # ------------------------------------------------------------------------------------------------uses the lane_image to do mask and get the borders/edges to get later the x1 ,x2,y1,y2
        frames = np.uint8(cv2.GaussianBlur(lane_image, (5, 5), 0))
        hsv = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)

        lower_y = np.array([40, 40, 40])
        upper_y = np.array([70, 255, 255])

        mask = cv2.inRange(hsv, lower_y, upper_y)
        edges = cv2.Canny(mask, 50, 150)  # Adjust Canny thresholds for better edge detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=80)  # Adjust Hough parameters
        x1_coordinates = []
        y1_coordinates = []

        # Loop over the lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Collect x1, y1 coordinates
                x1_coordinates.append(x1)
                x1_coordinates.append(x2)
                y1_coordinates.append(y1)
                y1_coordinates.append(y2)
                cv2.line(frames, (x1, y1), (x2, y2), (255, 255, 0), 5)
            # Create lines using the collected coordinates
            line1 = [np.min(x1_coordinates), np.min(y1_coordinates), np.max(x1_coordinates), np.max(y1_coordinates)]
        
            x1,x2= line1[0],line1[2]
            cv2.circle(frames, (int((np.mean(x1_coordinates))),int((line1[1]+line1[3])/2)), 2, (255,0,255), 2)  # Draw predicted position
            
            mean_y=int((np.mean(y1_coordinates)))
            mean_x=int((np.mean(x1_coordinates))) 

            centerframe_x=250
            centerframe_y=250
            distance =int(depth_array[centerframe_x,mean_y]-depth_array[mean_x,mean_y])# in millimeter
            # Example usage
            object_center = (mean_x, mean_y)  # Example object center coordinates (x, y)
            frame_center = (250, 250)   # Example frame center coordinates (x, y)
            fov_horizontal_deg = 86   # Horizontal field of view of the camera
            fov_vertical_deg = 57     # Vertical field of view of the camera
            frame_width = 500           # Width of the frame
            frame_height = 500          # Height of the frame

            horizontal_angle, vertical_angle = calculate_angle(object_center, frame_center, fov_horizontal_deg, fov_vertical_deg, frame_width, frame_height)
            print("Horizontal angle:", horizontal_angle)
            print("Vertical angle:", vertical_angle)
            if(mean_x < centerframe_x):
                print('no')
                cv2.putText(frames, 'go right cuz'+str(distance), (centerframe_x , centerframe_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 4)
            elif(mean_x > centerframe_x):
                print('hey')
                cv2.putText(frames, 'go left cuz'+str(distance), (centerframe_x, centerframe_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 4)

            # ------------------------------------------------------------------------------------------------

        return frames



subscriberName = 'camera_lane_detection'

lanes=lanes()

def Function():
    global frame ,video,result
    if frame is not None :
        print('h')
        frame = cv2.resize(frame, (500,500))
        cv2.circle(frame, (250,250), 2, (255, 0, 0), 2)  # Draw predicted position
        video=road_lines()
        result = cv2.bitwise_or(cv2.resize(frame, (500,500)),cv2.resize(video, (500,500)))

def callbackFunction_color(message):
    global color_frame ,frame
    bridgeObject = CvBridge()
    rospy.loginfo('received a color video')
    color_frame = bridgeObject.imgmsg_to_cv2(message)
    frame = color_frame.copy()
    Function()
    
def callbackFunction_depth(message):
    global depth_frame
    bridgeObject = CvBridge()
    rospy.loginfo('received a depth video')
    depth_frame = bridgeObject.imgmsg_to_cv2(message)

def callbackFunction_deptharray(message):
    global depth_array
    row=480
    col=640
    depth_array = np.asanyarray(message.data).reshape(row,col)

def display_frames():
    global color_frame, depth_frame,frame,video,result
    while not rospy.is_shutdown():
        if color_frame is not None:
            cv2.imshow('color', color_frame)
        if depth_frame is not None:
            cv2.imshow('depth', depth_frame)
        if frame is not None:
            cv2.imshow('detection', frame)    
        if video is not None:
            cv2.imshow('video', video)    
        if result is not None:
            cv2.imshow('result', result)  
        key = cv2.waitKey(10)
        if key ==ord('q'):
            break

rospy.init_node(subscriberName, anonymous=True)
rospy.Subscriber(topicName, Image, callbackFunction_color)
rospy.Subscriber(topicName_depth, Image, callbackFunction_depth)
rospy.Subscriber(deptharray_topicName, Float32MultiArray, callbackFunction_deptharray)
display_frames()
rospy.spin()
