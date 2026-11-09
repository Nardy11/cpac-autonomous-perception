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
from std_msgs.msg import Float32MultiArray
from ros_opencv.msg import custom_msg_stereo

subscriberName = 'camera_sensor_subscriber'
topicName = 'color_video_topic'
topicName_depth = 'depth_video_topic'
deptharray_topicName = 'depth_array_topic'
publishername = 'Stereo_published'

bridgeObject = CvBridge()

# Decreased resolution for faster processing
new_width = 300
new_height = 300

# Set desired frame rate
desired_fps = 10

# Create windows for color and depth frames
cv2.namedWindow('color', cv2.WINDOW_NORMAL)
cv2.namedWindow('depth', cv2.WINDOW_NORMAL)

color_frame = None
depth_frame = None

def callbackFunction_color(message):
    global color_frame
    bridgeObject = CvBridge()
    # rospy.loginfo('received a color video')
    color_frame = bridgeObject.imgmsg_to_cv2(message)

def callbackFunction_depth(message):
    global depth_frame
    bridgeObject = CvBridge()
    # rospy.loginfo('received a depth video')
    depth_frame = bridgeObject.imgmsg_to_cv2(message)

def callbackFunction_deptharray(message):
    global depth_array
    row=480
    col=640
    depth_array = np.asanyarray(message.data).reshape(row,col)
    # rospy.loginfo('received a depth array: %s',depth_array)

def display_frames():
    global color_frame, depth_frame
    while not rospy.is_shutdown():
        if color_frame is not None:
            cv2.imshow('color', color_frame)
        if depth_frame is not None:
            cv2.imshow('depth', depth_frame)
        cv2.waitKey(1)

# def callback(message):
    # rospy.loginfo('received a data customized: %s',message)




rospy.init_node(subscriberName, anonymous=True)
rospy.Subscriber(topicName, Image, callbackFunction_color)
rospy.Subscriber(topicName_depth, Image, callbackFunction_depth)
rospy.Subscriber(deptharray_topicName, Float32MultiArray, callbackFunction_deptharray)
# rospy.Subscriber(publishername, custom_msg_stereo, callback)

display_frames()

cv2.destroyAllWindows()