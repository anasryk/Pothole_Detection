import os
import sys
import time
import json
import shutil
import cv2
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(base_dir, "dataset", "yolo", "data.yaml")
    project_dir = os.path.join(base_dir, "runs", "detect")
    name = "train"

    print("=" * 60)
    print("STARTING YOLO11n TRAINING")
    print(f"Data: {data_yaml}")
    print(f"Project: {project_dir}")
    print(f"Run name: {name}")
    print("=" * 60)

    # Load YOLO11n model
    model = YOLO("yolo11n.pt")

    # Record training start time
    start_time = time.time()

    # Train model
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        patience=10,
        workers=2,
        device="cpu",
        project=project_dir,
        name=name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )

    training_time_sec = time.time() - start_time
    training_time_str = f"{int(training_time_sec // 3600)}h {int((training_time_sec % 3600) // 60)}m {int(training_time_sec % 60)}s"
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE! Total time: {training_time_str} ({training_time_sec:.2f}s)")
    print("=" * 60)

    # Best model path
    best_model_path = os.path.join(project_dir, name, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        # Fallback to last.pt if best.pt is not found
        best_model_path = os.path.join(project_dir, name, "weights", "last.pt")

    print(f"Best model path: {best_model_path}")

    # Evaluate on TEST set
    print("\n" + "=" * 60)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("=" * 60)

    best_model = YOLO(best_model_path)
    test_metrics = best_model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=8,
        device="cpu",
        project=project_dir,
        name="test_eval",
        exist_ok=True,
        plots=True
    )

    precision = float(test_metrics.box.mp) if hasattr(test_metrics.box, 'mp') else float(test_metrics.box.p[0])
    recall = float(test_metrics.box.mr) if hasattr(test_metrics.box, 'mr') else float(test_metrics.box.r[0])
    map50 = float(test_metrics.box.map50)
    map50_95 = float(test_metrics.box.map)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS:")
    print(f"  Precision:  {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:     {recall:.4f} ({recall*100:.2f}%)")
    print(f"  mAP50:      {map50:.4f} ({map50*100:.2f}%)")
    print(f"  mAP50-95:   {map50_95:.4f} ({map50_95*100:.2f}%)")
    print("=" * 60)

    # Save summary dictionary
    summary = {
        "training_completed_successfully": True,
        "model": "yolo11n",
        "epochs_requested": 50,
        "epochs_trained": getattr(results, "epoch", 50) + 1 if hasattr(results, "epoch") else 50,
        "training_time_seconds": round(training_time_sec, 2),
        "training_time_formatted": training_time_str,
        "best_model_path": best_model_path,
        "metrics_test": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4)
        },
        "plots_directory": os.path.join(project_dir, name),
        "test_eval_directory": os.path.join(project_dir, "test_eval")
    }

    summary_json_path = os.path.join(project_dir, "training_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_json_path}")

    # Generate sample prediction images on test set
    test_img_dir = os.path.join(base_dir, "dataset", "yolo", "images", "test")
    test_samples = sorted(os.listdir(test_img_dir))[:5]
    sample_paths = [os.path.join(test_img_dir, s) for s in test_samples]
    
    pred_dir = os.path.join(project_dir, "sample_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    best_model.predict(
        source=sample_paths,
        save=True,
        project=project_dir,
        name="sample_predictions",
        exist_ok=True,
        conf=0.25
    )
    print(f"Saved sample predictions to: {pred_dir}")

    # Also copy key plot files and predictions to artifact directory
    artifact_dir = r"C:\Users\Microsoft\.gemini\antigravity\brain\1c4c78a7-9b40-4be5-ba0b-897d54a3cf0c"
    os.makedirs(artifact_dir, exist_ok=True)
    for plot_file in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png", "results.png"]:
        src = os.path.join(project_dir, name, plot_file)
        if os.path.exists(src):
            dst = os.path.join(artifact_dir, f"yolo_{plot_file}")
            shutil.copy2(src, dst)
            print(f"Copied {plot_file} to artifact dir: {dst}")

if __name__ == "__main__":
    main()
