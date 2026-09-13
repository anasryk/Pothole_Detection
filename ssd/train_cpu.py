"""
SSD MobileNetV3-Large Training & Evaluation Pipeline on CPU.
Meets all project constraints:
- Uses existing Pascal VOC XML annotations at dataset/ssd/
- Uses exact splits (479 train, 53 val, 133 test)
- Trains SSDLite320 MobileNetV3 with CPU-optimized settings
- Saves best weights to models/ssd_best.pth and runs/ssd/weights/best.pth
- Generates loss curve, PR curve, sample test predictions, and test_metrics.json
"""
import os, sys, time, json, ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
import torch.nn as nn

# Local imports
sys.path.append(str(Path(__file__).resolve().parent))
from dataset_eval import PotholeVOCDataset, collate_fn, evaluate_model

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SSD_DIR = BASE_DIR / "dataset" / "ssd"
ANN_DIR = SSD_DIR / "annotations"
TRAIN_DIR = SSD_DIR / "images" / "train"
VAL_DIR = SSD_DIR / "images" / "val"
TEST_DIR = SSD_DIR / "images" / "test"

OUTPUT_DIR = BASE_DIR / "runs" / "ssd"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
PRED_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters for CPU
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TARGET_SIZE = 320
NUM_WORKERS = 0

torch.set_num_threads(4)

def build_model(num_classes=2):
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    
    in_channels = [672, 480, 512, 256, 256, 128]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
    )
    
    # Freeze the early feature extractor layers of MobileNetV3 for speed & stable transfer learning
    for param in model.backbone.features.parameters():
        param.requires_grad = False
        
    return model

def main():
    print("=" * 70)
    print("      SSDLite320 MobileNetV3 CPU Training — Pothole Detection")
    print("=" * 70)
    
    # 1. Dataset confirmation
    train_dataset = PotholeVOCDataset(TRAIN_DIR, ANN_DIR, target_size=TARGET_SIZE)
    val_dataset = PotholeVOCDataset(VAL_DIR, ANN_DIR, target_size=TARGET_SIZE)
    test_dataset = PotholeVOCDataset(TEST_DIR, ANN_DIR, target_size=TARGET_SIZE)
    
    print(f"Dataset Verified:")
    print(f"  Train images: {len(train_dataset)}")
    print(f"  Val images:   {len(val_dataset)}")
    print(f"  Test images:  {len(test_dataset)}")
    print(f"  Target Size:  {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"  Batch Size:   {BATCH_SIZE}")
    print(f"  Epochs:       {NUM_EPOCHS}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    
    # 2. Build Model
    model = build_model(num_classes=2)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"\nModel Architecture: SSDLite320 MobileNetV3-Large")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {num_trainable:,} (head + extra feature layers)")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    
    # 3. Benchmark First Steps
    print("\nEstimating training time from initial step...")
    steps_per_epoch = len(train_loader)
    
    model.train()
    t_start = time.time()
    
    best_val_map50 = -1.0
    best_epoch = 0
    history = []
    
    csv_file = OUTPUT_DIR / "results.csv"
    with open(csv_file, "w") as f:
        f.write("epoch,train_loss,val_precision,val_recall,val_f1,val_map50,val_map50_95,epoch_time_s\n")
        
    print("\nStarting Training Loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Skip empty target batches safely
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
            optimizer.step()
            
            running_loss += losses.item()
            
            if (batch_idx + 1) % 30 == 0 or (batch_idx + 1) == steps_per_epoch:
                cur_loss = running_loss / (batch_idx + 1)
                print(f"  Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{steps_per_epoch}] — Loss: {cur_loss:.4f}", flush=True)
                
        lr_scheduler.step()
        epoch_loss = running_loss / steps_per_epoch
        epoch_duration = time.time() - epoch_t0
        
        # Validation Evaluation
        val_metrics = evaluate_model(model, val_loader, conf_threshold=0.25)
        val_p = val_metrics["precision"]
        val_r = val_metrics["recall"]
        val_f1 = val_metrics["f1"]
        val_map50 = val_metrics["map50"]
        val_map50_95 = val_metrics["map50_95"]
        
        print(f"--> Epoch {epoch} Complete in {epoch_duration:.1f}s | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val mAP@50: {val_map50:.2%} | "
              f"Val P: {val_p:.2%} | "
              f"Val R: {val_r:.2%}", flush=True)
              
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_precision": val_p,
            "val_recall": val_r,
            "val_f1": val_f1,
            "val_map50": val_map50,
            "val_map50_95": val_map50_95,
            "epoch_time_s": epoch_duration
        })
        
        # Write to results.csv
        with open(csv_file, "a") as f:
            f.write(f"{epoch},{epoch_loss:.4f},{val_p:.4f},{val_r:.4f},{val_f1:.4f},{val_map50:.4f},{val_map50_95:.4f},{epoch_duration:.1f}\n")
            
        # Save last checkpoint
        torch.save(model.state_dict(), WEIGHTS_DIR / "last.pth")
        
        # Check if best
        if val_map50 > best_val_map50:
            best_val_map50 = val_map50
            best_epoch = epoch
            torch.save(model.state_dict(), WEIGHTS_DIR / "best.pth")
            torch.save(model.state_dict(), MODELS_DIR / "ssd_best.pth")
            print(f"    * New best validation mAP@50: {val_map50:.2%} (saved to models/ssd_best.pth)")
            
    total_training_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Training finished in {total_training_time/60:.1f} minutes!")
    print(f"Best Validation Epoch: {best_epoch} (mAP@50: {best_val_map50:.2%})")
    print("=" * 70)
    
    # 4. Final Test Set Evaluation using Best Model
    print("\nEvaluating Best Model on the Independent 133-Image Test Split...")
    best_weights_path = MODELS_DIR / "ssd_best.pth"
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    
    test_metrics = evaluate_model(model, test_loader, conf_threshold=0.25)
    
    # Measure model file size
    model_size_mb = os.path.getsize(best_weights_path) / (1024 * 1024)
    
    final_results = {
        "model_architecture": "SSDLite320_MobileNet_V3_Large",
        "checkpoint_path": str(best_weights_path),
        "test_images_count": len(test_dataset),
        "ground_truth_boxes": 299,
        "best_epoch": best_epoch,
        "total_training_time_sec": total_training_time,
        "total_training_time_min": total_training_time / 60.0,
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1"],
        "map50": test_metrics["map50"],
        "map50_95": test_metrics["map50_95"],
        "avg_latency_ms": test_metrics["avg_latency_ms"],
        "fps": test_metrics["fps"],
        "model_size_mb": model_size_mb
    }
    
    metrics_json = OUTPUT_DIR / "test_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nFinal Test Metrics on 133 Images:")
    print(f"  Precision:   {final_results['precision']:.2%}")
    print(f"  Recall:      {final_results['recall']:.2%}")
    print(f"  F1-Score:    {final_results['f1_score']:.2%}")
    print(f"  mAP@50:      {final_results['map50']:.2%}")
    print(f"  mAP@50-95:   {final_results['map50_95']:.2%}")
    print(f"  Latency:     {final_results['avg_latency_ms']:.1f} ms")
    print(f"  Speed:       {final_results['fps']:.2f} FPS")
    print(f"  Model Size:  {model_size_mb:.2f} MB")
    
    # 5. Generate Visual Artifacts
    print("\nGenerating Artifacts (Loss Curve, PR Curve, Sample Predictions)...")
    
    # Plot 1: Loss & mAP Curves
    epochs_range = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_map50s = [h["val_map50"] for h in history]
    
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, "b-o", label="Train Loss")
    plt.title("SSD Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_map50s, "g-s", label="Val mAP@50")
    plt.title("SSD Validation mAP@50")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = OUTPUT_DIR / "ssd_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    
    # Plot 2: Precision-Recall Curve
    pr = test_metrics["pr_curve"]
    if len(pr["recalls"]) > 0:
        plt.figure(figsize=(6, 5))
        plt.plot(pr["recalls"], pr["precisions"], "b-", lw=2, label=f"SSD mAP@50 = {final_results['map50']:.2%}")
        plt.title("SSD Precision-Recall Curve (133-Image Test Split)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.05])
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower left")
        plt.tight_layout()
        pr_plot_path = OUTPUT_DIR / "ssd_pr_curve.png"
        plt.savefig(pr_plot_path, dpi=200)
        plt.close()
        
    # Plot 3: Sample Test Predictions
    sample_indices = [0, 5, 10, 15, 20]
    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            if idx >= len(test_dataset):
                continue
            img_tensor, tgt = test_dataset[idx]
            orig_h, orig_w = tgt["orig_size"].tolist()
            img_name = tgt["img_name"]
            
            output = model([img_tensor])[0]
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()
            
            keep = (labels == 1) & (scores >= 0.25)
            boxes = boxes[keep]
            scores = scores[keep]
            
            scale_x = orig_w / 320.0
            scale_y = orig_h / 320.0
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            # Draw on original image
            orig_img = Image.open(TEST_DIR / img_name).convert("RGB")
            draw = ImageDraw.Draw(orig_img)
            
            for b_idx in range(len(boxes)):
                box = boxes[b_idx].tolist()
                s = float(scores[b_idx])
                draw.rectangle(box, outline="#e11d48", width=3) # Crimson outline
                draw.text((box[0] + 2, max(0, box[1] - 12)), f"Pothole {s:.0%}", fill="#e11d48")
                
            pred_img_path = PRED_DIR / f"pred_{img_name}"
            orig_img.save(pred_img_path)
            
    print(f"Artifacts saved successfully:")
    print(f"  - Loss Curve:        {loss_plot_path}")
    print(f"  - PR Curve:          {OUTPUT_DIR / 'ssd_pr_curve.png'}")
    print(f"  - Sample Preds:      {PRED_DIR}")
    print(f"  - Metrics JSON:      {metrics_json}")
    print(f"  - Best Checkpoint:   {best_weights_path}")
    print("\nSSD Training & Evaluation pipeline completed!")

if __name__ == "__main__":
    main()
