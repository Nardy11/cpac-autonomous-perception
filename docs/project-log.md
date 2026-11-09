# Project log: 12 weeks of the CPAC bachelor project

This summarises the weekly progress reports (`week1–5.docx`) and the code committed each week. The
reports themselves are not published: they mostly repeat the code with a line-by-line explanation.
Weeks 6–12 had no written report, so their entries come from the code alone.

| Week | What was built | Where it lives now |
|---|---|---|
| 1 | YOLOv5 webcam detector: box corners, centre, width and height printed and drawn per frame. | (only in report; superseded by week 3) |
| 2 | HSV colour-blob follower: green object → centroid → "MOVE LEFT / RIGHT / UP / DOWN" relative to the frame centre (first closed-loop "control" signal). Custom 13-class dataset defined (`last.pt`). | `detection/dataset.yaml` |
| 3 | YOLOv5s fine-tuned on the 13-class Roboflow set; monocular distance estimate from a linear fit of box diagonal (px) to measured distance (40–80 cm); CSV logging of every detection; FPS counter; annotated video output. | `detection/yolov5_detect_distance.py` |
| 4 | Same pipeline ported to YOLOv8n (Ultralytics 8.1.27). | `detection/yolov8_detect_distance.py` |
| 5 | Comparison model: TF2 Object Detection API, SSD-MobileNetV2-FPNLite 320×320, fine-tuned on the same 13 classes (checkpoints up to ckpt-51). | `detection/tf2/` |
| 6 | YOLOv8n exported to OpenVINO IR for faster CPU inference; TF2 SavedModel video runner. | `detection/yolov8n_openvino_detect.py`, `detection/tf2/ssd_mobilenet_video.py` |
| 7 | Multi-object tracking: Euclidean centroid tracker (IDs kept while the centre moves < 75 px). Two-webcam variant that matches people across both views for stereo triangulation. | `tracking/` |
| 8 | Motion prediction with a constant-velocity Kalman filter; Lucas–Kanade optical-flow trajectories; classical lane detection (Canny → ROI → Hough → averaged lane lines); CNN lane segmentation in Colab. | `tracking/track_kalman_optical_flow.py`, `lane_detection/` |
| 9 | CNN lane mask → lane centre → steering hint ("go left / go right" by pixel offset). First ROS 1 subscribers for detection and lane following. | `lane_detection/lane_cnn_steering.py`, `ros/` |
| 10 | Intel RealSense RGB-D: colour + depth published over ROS; detection node reads the depth at each person's box centre and computes the bearing from the camera FOV (86° × 57°). | `sensors/realsense_*.py`, `ros/.../camera_publisher.py`, `Camera_detection.py` |
| 11 | ZED stereo camera: distance from the point cloud and bearing per detection. Catkin package `ros_opencv` with custom messages (`custom_msg_stereo`, `custom_msg_monocular`) so other nodes and vehicles get object name, timestamp, distance and angle. Monocular node fuses the camera bearing with the LiDAR range at that angle. Flask MJPEG stream of the camera topic. | `sensors/zed_detect_distance.py`, `ros/ros_opencv/` |
| 12 | LiDAR (`/scan`): live polar plot in pygame, and a sender that pushes each scan to a Firebase Realtime Database under the car's id, so other vehicles and a dashboard can read it (the cooperative part). | `sensors/lidar_*.py` |

## Known issues kept as-is

The clean-up only moved hard-coded paths and secrets into CLI flags and environment variables, so
the original logic is untouched, including these quirks:

* The monocular distance (`coff = np.polyfit(...)`) was fitted to 5 hand measurements at 40–80 cm
  with one camera; it is not valid outside that range or for other cameras.
* Several scripts re-export `yolov8n.pt` to OpenVINO at start-up (`model.export(format='openvino')`).
  That is slow, but harmless.
* `tracking/track_two_cameras.py` imports a `triangulation` module (`find_depth`) that was not in
  the upload.
* The tracking scripts keep only the `person` class (`cls == 0`) before tracking.
* In the RealSense node, `depth_array[center[0], center[1]]` indexes (x, y) into a (row, col) array, so
  rows and columns are swapped. A rewrite would fix it; this repo keeps the original.
