#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

publishername = 'camera_sensor_publisher1'
topicName = 'video_topic'

rospy.init_node(publishername, anonymous=True)
publisher = rospy.Publisher(topicName, Image, queue_size=1)  # Decreased queue size
rate = rospy.Rate(300)

videoCaptureObject = cv2.VideoCapture(0)
bridgeObject = CvBridge()

# Decreased resolution for faster processing
new_width = 250
new_height = 250

# Set desired frame rate
desired_fps = 9
videoCaptureObject.set(cv2.CAP_PROP_FPS, desired_fps)

while not rospy.is_shutdown():
    returnValue, capturedFrame = videoCaptureObject.read()
    if returnValue:
        # Resize captured frame
        resized_frame = cv2.resize(capturedFrame, (new_width, new_height))
        
        # Convert to grayscale for faster processing
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale frame to ROS Image message
        imageToTransmit = bridgeObject.cv2_to_imgmsg(resized_frame, encoding="bgr8") #"bgr8" #"mono8"
        
        # Publish the image
        publisher.publish(imageToTransmit)
    
    rate.sleep()
