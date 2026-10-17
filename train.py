"""Fine-tune YOLO11n on the pothole dataset, then validate and time inference.

    python download_data.py
    python train.py
"""

import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SEED = 42


def resolved_data_yaml():
    # Ultralytics resolves relative dataset paths against its global datasets_dir,
    # so write a copy with an absolute path instead.
    cfg = yaml.safe_load((ROOT / "data.yaml").read_text())
    cfg["path"] = str(ROOT / cfg["path"])
    out = ROOT / "runs" / "data_resolved.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg))
    return str(out)


def main():
    data = resolved_data_yaml()
    model = YOLO("yolo11n.pt")
    model.train(
        data=data,
        imgsz=640,
        epochs=30,
        batch=-1,  # auto batch size
        seed=SEED,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="pothole",
        exist_ok=True,
    )

    run = ROOT / "runs" / "pothole"
    best = YOLO(run / "weights" / "best.pt")
    metrics = {}
    for split in ("val", "test"):
        m = best.val(data=data, split=split, imgsz=640, project=str(ROOT / "runs"), name=f"val_{split}", exist_ok=True)
        metrics[split] = {
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50-95": round(float(m.box.map), 4),
        }

    # Inference speed on a few test images (first image excluded as warm-up).
    images = sorted((ROOT / "datasets" / "pothole" / "test" / "images").glob("*"))[:11]
    results = [best.predict(str(p), imgsz=640, verbose=False)[0] for p in images]
    speeds = [r.speed for r in results[1:]]
    metrics["speed_ms_per_image"] = {
        k: round(sum(s[k] for s in speeds) / len(speeds), 2) for k in ("preprocess", "inference", "postprocess")
    }
    metrics["speed_images"] = len(speeds)

    # Copy committed artifacts out of the gitignored runs/ folder.
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(run / "weights" / "best.pt", ROOT / "weights" / "best.pt")
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    for f in ("results.csv", "results.png", "confusion_matrix.png", "BoxPR_curve.png", "val_batch0_pred.jpg"):
        if (run / f).exists():
            shutil.copy(run / f, res / f)
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
