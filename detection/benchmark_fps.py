"""Benchmark the custom YOLOv8n on CPU: PyTorch vs ONNX Runtime vs OpenVINO.

Runs the same N frames of a video through each backend (Ultralytics predictor, so
pre/post-processing is identical) and reports end-to-end FPS and the model's own
inference time.

    python detection/benchmark_fps.py --video detection_video_blurred.mp4 \
        --models weights/yolov8n_custom_last.pt weights/yolov8n_custom_last.onnx \
                 weights/yolov8n_custom_last_openvino_model
"""
import argparse
import platform
import statistics
import time

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--video", required=True)
parser.add_argument("--models", nargs="+", required=True, help=".pt, .onnx or *_openvino_model dir")
parser.add_argument("--frames", type=int, default=100)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
frames = []
while len(frames) < args.frames + args.warmup:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()

print(f"CPU: {platform.processor()} | {platform.system()} {platform.release()} | "
      f"{len(frames) - args.warmup} frames {frames[0].shape[1]}x{frames[0].shape[0]} -> imgsz {args.imgsz}")
print(f"{'backend':<45} {'FPS (end-to-end)':>17} {'inference ms (median)':>22}")
for path in args.models:
    model = YOLO(path, task="detect")
    for f in frames[: args.warmup]:
        model(f, imgsz=args.imgsz, device="cpu", verbose=False)
    infer = []
    t0 = time.perf_counter()
    for f in frames[args.warmup:]:
        r = model(f, imgsz=args.imgsz, device="cpu", verbose=False)[0]
        infer.append(r.speed["inference"])
    dt = time.perf_counter() - t0
    n = len(frames) - args.warmup
    print(f"{path:<45} {n / dt:>17.1f} {statistics.median(infer):>22.1f}")
