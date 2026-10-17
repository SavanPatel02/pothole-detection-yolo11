# Pothole Detection with YOLO11n

Detect potholes in road images with a fine-tuned YOLO11n model, plus a Gradio demo.

Work in progress.

## Run the demo

```bash
pip install -r requirements.txt
python app.py   # opens http://127.0.0.1:7860
```

Upload a road image, adjust the confidence slider, and the app draws a box around each detected pothole using `weights/best.pt`.

## License

MIT, see [LICENSE](LICENSE).
