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

model =keras.models.load_model(os.environ.get('LANE_MODEL', 'model.h5'))

class lanes():
  def __init__(self):
    self.recent_fit =[]
    self.avg_fit =[]


def road_lines(image):
    # Resize the image to the expected shape
    smallimg = cv2.resize(image, (160, 80))  # Corrected resizing dimensions
    smallimg = np.array(smallimg)
    smallimg = smallimg[None,:,:,:]

    # Assuming 'model' is defined and loaded somewhere in your code
    prediction = model.predict(smallimg)[0] * 255
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
        if(int((np.mean(x1_coordinates))) < (frames.shape[0]/2)):
            print('no')
            cv2.putText(frames, 'go right cuz'+str((frames.shape[0]/2)-(int(np.mean(x1_coordinates)))), (int(frames.shape[0]/2), int(frames.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 4)
        elif(int((np.mean(x1_coordinates))) > (frames.shape[0]/2)):
            print('hey')
            cv2.putText(frames, 'go left cuz'+str(int(np.mean(x1_coordinates))-(frames.shape[0]/2)), (int(frames.shape[0]/2), int(frames.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 4)


    cv2.imshow("image",image)
        # ------------------------------------------------------------------------------------------------

    # Specify the data type of the output array

    cv2.waitKey(1)
    return frames



subscriberName = 'camera_lane_detection'
topicName = 'video_topic'

lanes=lanes()

def callbackFunction(message):
    bridgeObject = CvBridge()
    rospy.loginfo('received a video')
    convertedFrame = bridgeObject.imgmsg_to_cv2(message)
    frame = cv2.resize(convertedFrame, (500,500))
    cv2.circle(frame, (250,250), 2, (255, 0, 0), 2)  # Draw predicted position
    video=road_lines(cv2.resize(frame, (500,500)))
    cv2.imshow("framed", video)
    result = cv2.bitwise_or(cv2.resize(frame, (500,500)),cv2.resize(video, (500,500)))
    cv2.imshow("result", result)
    cv2.waitKey(1)
    cv2.imshow('YOLO_V8',cv2.resize(convertedFrame,(500,500)))
    if cv2.waitKey(10) == ord('q'):
        cv2.destroyAllWindows()

rospy.init_node(subscriberName, anonymous=True)
# publisher = rospy.Publisher(topicSend, Image, queue_size=0)  # Decreased queue size
rate = rospy.Rate(300)
rospy.Subscriber(topicName, Image, callbackFunction)
rospy.spin()