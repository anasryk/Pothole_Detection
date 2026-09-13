"""
Full SSD Training and Evaluation script.
Trains SSDLite320 MobileNetV3 on CPU with tracking, metrics, and artifact generation.
"""
import os, sys, time, json, math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
import torch.nn as nn

# Set CPU threads
torch.set_num_threads(4)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class PotholeVOCDataset(Dataset):
    def __init__(self, images_dir: Path, annotations_dir: Path, target_size=320):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.target_size = target_size
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        xml_path = self.annotations_dir / f"{img_path.stem}.xml"
        
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        boxes = []
        labels = []
        if xml_path.exists():
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name.lower() == "pothole":
                    bnd = obj.find("bndbox")
                    xmin = float(bnd.find("xmin").text)
                    ymin = float(bnd.find("ymin").text)
                    xmax = float(bnd.find("xmax").text)
                    ymax = float(bnd.find("ymax").text)
                    xmin = max(0.0, min(xmin, float(orig_w)))
                    ymin = max(0.0, min(ymin, float(orig_h)))
                    xmax = max(xmin + 1.0, min(xmax, float(orig_w)))
                    ymax = max(ymin + 1.0, min(ymax, float(orig_h)))
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1) # pothole
                    
        # Resize image
        image_resized = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        scale_x = self.target_size / orig_w
        scale_y = self.target_size / orig_h
        
        scaled_boxes = []
        for b in boxes:
            scaled_boxes.append([
                b[0] * scale_x,
                b[1] * scale_y,
                b[2] * scale_x,
                b[3] * scale_y
            ])
            
        img_tensor = F.to_tensor(image_resized)
        
        if len(scaled_boxes) > 0:
            boxes_tensor = torch.as_tensor(scaled_boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "orig_boxes": torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4)),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "img_name": img_path.name
        }
        return img_tensor, target

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# -----------------------------------------------------------------------------
# Evaluation Helper (Standard COCO / VOC mAP@50 and mAP@50-95)
# -----------------------------------------------------------------------------
def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)

def compute_ap(recalls, precisions):
    # Standard 101-point or continuous interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_model(model, data_loader, conf_threshold=0.25, iou_threshold=0.45):
    """
    Evaluates model on given data_loader.
    Computes precision, recall, f1, mAP@50, mAP@50-95, and average latency.
    """
    model.eval()
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    
    latencies = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            t0 = time.time()
            outputs = model(images)
            lat = (time.time() - t0) * 1000 / len(images)
            latencies.append(lat)
            
            for i in range(len(images)):
                out = outputs[i]
                orig_h, orig_w = targets[i]["orig_size"].tolist()
                scale_x = orig_w / 320.0
                scale_y = orig_h / 320.0
                
                boxes = out["boxes"].cpu()
                scores = out["scores"].cpu()
                labels = out["labels"].cpu()
                
                # Keep pothole predictions (class 1)
                pothole_mask = (labels == 1)
                boxes = boxes[pothole_mask]
                scores = scores[pothole_mask]
                
                # Scale boxes back to original image coordinates for exact ground-truth comparison
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                
                all_pred_boxes.append(boxes)
                all_pred_scores.append(scores)
                all_gt_boxes.append(targets[i]["orig_boxes"])
                
    avg_latency = float(np.mean(latencies))
    
    # Calculate mAP across IoU thresholds 0.50 to 0.95
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    aps = []
    
    # For metrics at IoU 0.50:
    metrics_at_50 = {}
    
    for iou_thresh in iou_thresholds:
        # Collect matches across all images
        n_gts = sum(len(gt) for gt in all_gt_boxes)
        
        all_matches = [] # (score, is_tp)
        for i in range(len(all_gt_boxes)):
            preds = all_pred_boxes[i]
            scores = all_pred_scores[i]
            gts = all_gt_boxes[i]
            
            if len(preds) == 0:
                continue
                
            # Sort predictions by score descending
            order = scores.argsort(descending=True)
            preds = preds[order]
            scores = scores[order]
            
            if len(gts) == 0:
                for s in scores:
                    all_matches.append((float(s), 0))
                continue
                
            ious = box_iou(preds, gts)
            detected = set()
            for p_idx in range(len(preds)):
                s = float(scores[p_idx])
                max_iou, gt_idx = ious[p_idx].max(dim=0)
                max_iou = float(max_iou)
                gt_idx = int(gt_idx)
                
                if max_iou >= iou_thresh and gt_idx not in detected:
                    detected.add(gt_idx)
                    all_matches.append((s, 1))
                else:
                    all_matches.append((s, 0))
                    
        if len(all_matches) == 0:
            aps.append(0.0)
            continue
            
        all_matches.sort(key=lambda x: x[0], reverse=True)
        scores_arr = np.array([m[0] for m in all_matches])
        tp_arr = np.array([m[1] for m in all_matches])
        fp_arr = 1 - tp_arr
        
        tp_cumsum = np.cumsum(tp_arr)
        fp_cumsum = np.cumsum(fp_arr)
        
        recalls = tp_cumsum / max(1, n_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        
        if abs(iou_thresh - 0.50) < 1e-4:
            # Metrics at default conf_threshold
            valid_mask = scores_arr >= conf_threshold
            if np.sum(valid_mask) > 0:
                tp_val = tp_cumsum[np.where(valid_mask)[0][-1]]
                fp_val = fp_cumsum[np.where(valid_mask)[0][-1]]
                p = tp_val / (tp_val + fp_val)
                r = tp_val / n_gts
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            else:
                p, r, f1 = 0.0, 0.0, 0.0
            metrics_at_50 = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "map50": float(ap),
                "recalls": recalls,
                "precisions": precisions,
                "scores": scores_arr
            }
            
    map50 = aps[0]
    map50_95 = float(np.mean(aps))
    
    return {
        "precision": metrics_at_50.get("precision", 0.0),
        "recall": metrics_at_50.get("recall", 0.0),
        "f1": metrics_at_50.get("f1", 0.0),
        "map50": map50,
        "map50_95": map50_95,
        "avg_latency_ms": avg_latency,
        "fps": 1000.0 / avg_latency if avg_latency > 0 else 0.0,
        "pr_curve": {
            "recalls": metrics_at_50.get("recalls", np.array([])).tolist(),
            "precisions": metrics_at_50.get("precisions", np.array([])).tolist()
        }
    }

if __name__ == "__main__":
    print("Helper script ready.")
