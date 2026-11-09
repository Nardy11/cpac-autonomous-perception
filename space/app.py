"""Gradio app for a Hugging Face Space: CPAC road-obstacle detection (custom YOLOv8n vs COCO YOLOv8n).

Weights are downloaded from the GitHub release on first run.
"""
import os
import urllib.request

import gradio as gr
from ultralytics import YOLO

RELEASE = "https://github.com/Nardy11/cpac-autonomous-perception/releases/download/v1.0.0/"
WEIGHTS = {
    "Fine-tuned YOLOv8n (13 classes)": "yolov8n_custom_last.pt",
    "COCO YOLOv8n baseline (80 classes)": "yolov8n_coco.pt",
}
_models = {}


def get_model(label):
    name = WEIGHTS[label]
    if name not in _models:
        if not os.path.exists(name):
            urllib.request.urlretrieve(RELEASE + name, name)
        _models[name] = YOLO(name)
    return _models[name]


def detect(image, model_label, conf):
    if image is None:
        return None, []
    r = get_model(model_label)(image, conf=conf, verbose=False)[0]
    rows = [[r.names[int(c)], round(float(s), 3)] for c, s in zip(r.boxes.cls, r.boxes.conf)]
    return r.plot()[:, :, ::-1], rows


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Road image"),
        gr.Radio(list(WEIGHTS), value=list(WEIGHTS)[0], label="Model"),
        gr.Slider(0.05, 0.9, value=0.15, step=0.05, label="Confidence threshold"),
    ],
    outputs=[gr.Image(label="Detections"), gr.Dataframe(headers=["class", "confidence"], label="Objects")],
    title="CPAC: road-obstacle detection",
    description="YOLOv8n fine-tuned on the Roboflow 'car-obsticles' dataset (CC BY 4.0) vs the stock COCO YOLOv8n. "
                "Code: https://github.com/Nardy11/cpac-autonomous-perception",
)

if __name__ == "__main__":
    demo.launch()
