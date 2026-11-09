#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2
from std_msgs.msg import Float32MultiArray
publishername = 'camera_sensor_publisher'
color_topicName = 'color_video_topic'
depth_topicName = 'depth_video_topic'
deptharray_topicName = 'depth_array_topic'
pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)

pipe.start(cfg)

rospy.init_node(publishername, anonymous=True)
color_publisher = rospy.Publisher(color_topicName, Image, queue_size=1)
depth_publisher = rospy.Publisher(depth_topicName, Image, queue_size=1)
deptharray_publisher = rospy.Publisher(deptharray_topicName, Float32MultiArray, queue_size=1)

rate = rospy.Rate(300)


bridgeObject = CvBridge()

# Decreased resolution for faster processing
new_width = 500
new_height = 500

while not rospy.is_shutdown():
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Resize captured frames
    resized_color_frame = cv2.resize(color_image, (new_width, new_height))
    resized_depth_frame = cv2.resize(depth_cm, (new_width, new_height))

    # Convert resized frames to ROS Image messages
    color_imageToTransmit = bridgeObject.cv2_to_imgmsg(resized_color_frame, encoding="bgr8")
    depth_imageToTransmit = bridgeObject.cv2_to_imgmsg(resized_depth_frame, encoding="bgr8")
    ros_msg =Float32MultiArray()
    ros_msg.data=depth_image.flatten().tolist()
    # Publish the images
    color_publisher.publish(color_imageToTransmit)
    depth_publisher.publish(depth_imageToTransmit)
    deptharray_publisher.publish(ros_msg)
    rate.sleep()

pipe.stop()
