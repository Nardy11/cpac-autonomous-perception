#!/usr/bin/env python
import rospy
import random
import time
import os
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, Response
from flask_cors import CORS
import cv2
import numpy as np
import threading
from sensor_msgs.msg import LaserScan

print("firebase_admin imported successfully")

Car_ID = os.environ.get('CAR_ID', '1')

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Initialize CvBridge and threading lock
lock = threading.Lock()
Lidar_data=LaserScan()
# Function to run the Flask server
def run_flask_server():
    app.run(host='0.0.0.0', port=3000)

# Initialize Firebase Admin SDK with service account key
cred = credentials.Certificate(os.environ['FIREBASE_CREDENTIALS'])  # path to your service-account JSON (never commit it)
firebase_admin.initialize_app(cred, {
    'databaseURL': os.environ['FIREBASE_DB_URL']  # e.g. https://<project>-default-rtdb.firebaseio.com/
})

# ROS callback function for sensor data
def lidarCallback(data):
    global Lidar_data
    Lidar_data=data

def send_Lidar_data(Car_ID):
    global Lidar_data
    data = {
        'angle_increment': Lidar_data.angle_increment,
        'angle_max': Lidar_data.angle_max,
        'angle_min': Lidar_data.angle_min,
        'range_max': Lidar_data.range_max,
        'range_min':Lidar_data.range_min,
        'ranges': Lidar_data.ranges
    }
    ref = db.reference(f'cars/{Car_ID}/lidarData')
    ref.update(data)
    return data

def main():

    rate = rospy.Rate(300)  # Publish data at 300 Hz
    
    while not rospy.is_shutdown():
        # Write data to the Firebase database
        send_Lidar_data(Car_ID)

        rate.sleep()

if __name__ == "__main__":
    # Start the Flask server in a separate thread
    threading.Thread(target=run_flask_server).start()

    # Start ROS node in a separate thread
    rospy.init_node('Lidar_Firebase')
    #threading.Thread(target=lambda: rospy.init_node('image_listener', disable_signals=True)).start()

    # Subscribe to the ROS topic for images
    rospy.Subscriber('/scan', LaserScan, lidarCallback)
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass
