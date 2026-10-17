"""Gradio demo: upload a road image, get potholes boxed.

    python app.py
"""

from pathlib import Path

import gradio as gr
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "weights" / "best.pt"
model = YOLO(MODEL_PATH)


def detect(image, conf):
    if image is None:
        return None, "Upload an image first."
    result = model.predict(image, imgsz=640, conf=conf, verbose=False)[0]
    n = len(result.boxes)
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    return annotated, f"{n} pothole{'s' if n != 1 else ''} detected (confidence >= {conf:.2f})"


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Road image"),
        gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=[gr.Image(type="numpy", label="Detections"), gr.Textbox(label="Summary")],
    title="Pothole Detection (YOLO11n)",
    description="Upload a road photo. A YOLO11n model fine-tuned on 465 pothole images draws a box around each pothole.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
