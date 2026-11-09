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
from geometry_msgs.msg import Twist
from ros_opencv.msg import custom_msg_monocular
publishername = 'Monocular_published'

subscriberName = 'camera_sensor_subscriber'
topicName = 'video_topic'

topicSend = 'frame_send'

bridgeObject = CvBridge()

# Decreased resolution for faster processing
new_width = 300
new_height = 300

# Set desired frame rate
desired_fps = 10

def callbackFunction(message):
    bridgeObject = CvBridge()
    #rospy.loginfo('received a video')
    convertedFrame = bridgeObject.imgmsg_to_cv2(message)
    cv2.imshow('YOLO_V8',cv2.resize(convertedFrame,(500,500)))
# --------------------------------------------------------------------------------------------------------------publishing part 
    resized_frame = cv2.resize(convertedFrame, (new_width, new_height))
    
    #transform the image from cv2 readable image to bridge transferable image 
    imageToTransmit = bridgeObject.cv2_to_imgmsg(resized_frame, encoding="bgr8") #"bgr8" #"mono8"
        
    # Publish the image
    publisher.publish(imageToTransmit)
# --------------------------------------------------------------------------------------------------------------
        #to put an ending sign for each iteration
    if cv2.waitKey(10) == ord('q'):
        cv2.destroyAllWindows()
        


#def callback(message):
    #rospy.loginfo('received a data customized: %s',message)



rospy.init_node(subscriberName, anonymous=True)
publisher = rospy.Publisher(topicSend, Image, queue_size=0)  # Decreased queue size
rate = rospy.Rate(300)
rospy.Subscriber(topicName, Image, callbackFunction)
#rospy.Subscriber(publishername, custom_msg_monocular, callback)

rospy.spin()
