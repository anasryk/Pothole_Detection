import os
import sys
import time
import json
import shutil
import torch
from ultralytics import YOLO

def main():
    # Set thread count for optimized CPU inference/training
    torch.set_num_threads(4)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(base_dir, "dataset", "yolo", "data.yaml")
    project_dir = os.path.join(base_dir, "runs", "detect")
    run_name = "train_cpu"

    print("=" * 65)
    print("STARTING CPU-OPTIMIZED YOLO11n TRAINING")
    print(f"Dataset YAML:  {data_yaml}")
    print(f"Project Dir:   {project_dir}")
    print(f"Run Name:      {run_name}")
    print(f"PyTorch Threads: {torch.get_num_threads()}")
    print("=" * 65)

    # 1. Load pretrained YOLO11n model
    pretrained_weights = os.path.join(base_dir, "yolo11n.pt")
    if not os.path.exists(pretrained_weights):
        pretrained_weights = "yolo11n.pt"
    model = YOLO(pretrained_weights)

    # 2. Train with CPU-optimized configuration
    t_train_start = time.time()
    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=416,
        batch=4,
        patience=5,
        workers=0,
        device="cpu",
        project=project_dir,
        name=run_name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )
    t_train_end = time.time()
    training_time_sec = t_train_end - t_train_start
    mins = int(training_time_sec // 60)
    secs = int(training_time_sec % 60)
    training_time_str = f"{mins}m {secs}s ({training_time_sec:.2f}s)"

    print("\n" + "=" * 65)
    print(f"TRAINING COMPLETED IN: {training_time_str}")
    print("=" * 65)

    # 3. Locate best model weights
    train_dir = os.path.join(project_dir, run_name)
    best_model_path = os.path.join(train_dir, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(train_dir, "weights", "last.pt")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model weights not found at {best_model_path}")

    print(f"\n[BEST MODEL FOUND] -> {best_model_path}")

    # Determine best epoch if available
    best_epoch = None
    results_csv = os.path.join(train_dir, "results.csv")
    if os.path.exists(results_csv):
        import pandas as pd
        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            # Find epoch with highest mAP50 or mAP50-95
            map50_col = [c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()]
            if map50_col:
                best_idx = df[map50_col[0]].idxmax()
                best_epoch = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else int(best_idx + 1)
        except Exception as e:
            print(f"Could not parse best epoch from CSV: {e}")

    # 4. Evaluate BEST model on the TEST dataset
    print("\n" + "=" * 65)
    print("EVALUATING BEST MODEL ON TEST SPLIT")
    print("=" * 65)

    test_eval_dir = os.path.join(project_dir, "test_eval")
    best_model = YOLO(best_model_path)
    test_results = best_model.val(
        data=data_yaml,
        split="test",
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        project=project_dir,
        name="test_eval",
        exist_ok=True,
        plots=True
    )

    p = float(test_results.box.mp) if hasattr(test_results.box, 'mp') else float(test_results.box.p[0])
    r = float(test_results.box.mr) if hasattr(test_results.box, 'mr') else float(test_results.box.r[0])
    map50 = float(test_results.box.map50)
    map50_95 = float(test_results.box.map)
    f1 = 2 * (p * r) / (p + r + 1e-16)

    speeds = getattr(test_results, "speed", {})
    inference_ms = speeds.get("inference", None)

    print("\n" + "=" * 65)
    print("FINAL TEST SPLIT RESULTS:")
    print(f"  Precision:         {p:.4f} ({p*100:.2f}%)")
    print(f"  Recall:            {r:.4f} ({r*100:.2f}%)")
    print(f"  F1-Score:          {f1:.4f} ({f1*100:.2f}%)")
    print(f"  mAP50:             {map50:.4f} ({map50*100:.2f}%)")
    print(f"  mAP50-95:          {map50_95:.4f} ({map50_95*100:.2f}%)")
    if inference_ms is not None:
        print(f"  Inference Speed:   {inference_ms:.2f} ms/image")
    print(f"  Best Epoch:        {best_epoch}")
    print(f"  Training Time:     {training_time_str}")
    print(f"  Best Model Path:   {best_model_path}")
    print("=" * 65)

    # 5. Generate sample predictions
    test_img_dir = os.path.join(base_dir, "dataset", "yolo", "images", "test")
    test_samples = sorted(os.listdir(test_img_dir))[:5]
    sample_img_paths = [os.path.join(test_img_dir, s) for s in test_samples]

    sample_pred_dir = os.path.join(project_dir, "sample_predictions")
    best_model.predict(
        source=sample_img_paths,
        save=True,
        project=project_dir,
        name="sample_predictions",
        exist_ok=True,
        conf=0.25,
        imgsz=416
    )
    print(f"\nSaved sample predictions to: {sample_pred_dir}")

    # 6. Save comprehensive summary JSON
    summary = {
        "training_completed_successfully": True,
        "model": "yolo11n",
        "epochs": 10,
        "best_epoch": best_epoch,
        "imgsz": 416,
        "batch": 4,
        "device": "cpu",
        "best_model_path": os.path.abspath(best_model_path),
        "training_time_seconds": round(training_time_sec, 2),
        "training_time_formatted": training_time_str,
        "metrics_test": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "inference_ms": round(inference_ms, 2) if inference_ms is not None else None,
            "speeds": speeds
        },
        "artifacts": {
            "results_png": os.path.join(train_dir, "results.png"),
            "confusion_matrix": os.path.join(train_dir, "confusion_matrix.png"),
            "PR_curve": os.path.join(train_dir, "BoxPR_curve.png") if os.path.exists(os.path.join(train_dir, "BoxPR_curve.png")) else os.path.join(train_dir, "PR_curve.png"),
            "F1_curve": os.path.join(train_dir, "BoxF1_curve.png") if os.path.exists(os.path.join(train_dir, "BoxF1_curve.png")) else os.path.join(train_dir, "F1_curve.png"),
            "sample_predictions_dir": sample_pred_dir,
            "best_pt": os.path.abspath(best_model_path)
        }
    }

    summary_file = os.path.join(project_dir, "yolo_evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to: {summary_file}")

    # 7. Copy artifacts to current brain artifact directory
    artifact_dir = r"C:\Users\Microsoft\.gemini\antigravity-ide\brain\59c1e2ce-0c73-4804-8d6f-2672e8ec50c9"
    os.makedirs(artifact_dir, exist_ok=True)
    plots_to_copy = [
        ("results.png", "yolo_results.png"),
        ("confusion_matrix.png", "yolo_confusion_matrix.png"),
        ("confusion_matrix_normalized.png", "yolo_confusion_matrix_normalized.png"),
        ("BoxPR_curve.png", "yolo_PR_curve.png"),
        ("PR_curve.png", "yolo_PR_curve.png"),
        ("BoxF1_curve.png", "yolo_F1_curve.png"),
        ("F1_curve.png", "yolo_F1_curve.png")
    ]
    for src_name, dst_name in plots_to_copy:
        src_path = os.path.join(train_dir, src_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(artifact_dir, dst_name)
            shutil.copy2(src_path, dst_path)
            print(f"Copied plot {src_name} -> {dst_path}")

    # Copy sample predictions
    for sample_file in os.listdir(sample_pred_dir):
        if sample_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy2(os.path.join(sample_pred_dir, sample_file), os.path.join(artifact_dir, f"yolo_pred_{sample_file}"))

    print("\nALL EVALUATION TASKS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
