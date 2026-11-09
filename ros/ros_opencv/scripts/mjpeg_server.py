#!/usr/bin/env python3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
import cv2
import numpy as np
from flask import Flask, Response
from flask_cors import CORS
import threading

subscriberName = 'image_listener'
topicName = 'video_topic'
app = Flask(__name__)
CORS(app)

bridgeObject = CvBridge()
lock = threading.Lock()

color_frame = None

def callbackFunction_color(message):
    global color_frame
    bridgeObject = CvBridge()
    color_frame = bridgeObject.imgmsg_to_cv2(message)

def generate_frames():
    global color_frame
    while True:
        if color_frame is not None:
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', color_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask_server():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    threading.Thread(target=lambda:rospy.init_node(subscriberName, disable_signals=True)).start()
    rospy.Subscriber(topicName, Image, callbackFunction_color)
    threading.Thread(target=run_flask_server).start()
