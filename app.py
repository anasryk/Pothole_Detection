import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import torch
from functools import partial
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo_best.pt"
YOLO_FALLBACK_PATH = BASE_DIR / "runs" / "detect" / "train_cpu" / "weights" / "best.pt"

SSD_MODEL_PATH = BASE_DIR / "models" / "ssd_best.pth"
SSD_FALLBACK_PATH = BASE_DIR / "runs" / "ssd" / "weights" / "best.pth"

TEST_IMAGES_DIR = BASE_DIR / "dataset" / "yolo" / "images" / "test"

# Benchmark Test Set Metrics (133 test images, 299 ground-truth potholes)
YOLO_METRICS = {
    "precision": "74.03%",
    "recall": "62.54%",
    "f1_score": "67.80%",
    "map50": "70.94%",
    "map50_95": "43.36%",
    "inference_speed": "~101 ms",
    "fps": "~9.25 FPS",
    "model_size": "5.18 MB",
    "training_time": "~21 min",
    "best_epoch": "Epoch 10"
}

SSD_METRICS = {
    "precision": "20.47%",
    "recall": "46.15%",
    "f1_score": "28.37%",
    "map50": "32.92%",
    "map50_95": "13.42%",
    "inference_speed": "126.0 ms",
    "fps": "~7.93 FPS",
    "model_size": "8.70 MB",
    "training_time": "18.0 min",
    "best_epoch": "Epoch 3"
}

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pothole Detection | YOLO11n vs SSD",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Adaptive SaaS aesthetic (Supports both Light and Dark mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header typography */
    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
    }
    
    /* Metric summary cards */
    .metric-box {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: var(--primary-color);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color);
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color);
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 0.5rem;
    }
    
    /* Winner Banner */
    .winner-box {
        background-color: rgba(22, 163, 74, 0.1);
        border: 1px solid rgba(22, 163, 74, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1.5rem;
    }
    
    /* Upload Area Empty State */
    .upload-empty {
        background-color: var(--secondary-background-color);
        border: 2px dashed rgba(128, 128, 128, 0.4);
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
        transition: all 0.2s ease;
    }
    .upload-empty:hover {
        border-color: var(--primary-color);
        background-color: rgba(128, 128, 128, 0.05);
    }
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    .upload-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }
    .upload-sub {
        font-size: 0.95rem;
        color: var(--text-color);
        opacity: 0.7;
    }
    
    /* Model Tags */
    .model-tag-yolo {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .model-tag-ssd {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    /* Footer */
    .app-footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center;
        color: var(--text-color);
        opacity: 0.6;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model Loaders (Cached with Warmup)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    """
    Loads and caches the trained YOLO11n model.
    Runs a tiny warmup inference to pre-allocate buffers.
    """
    try:
        model = YOLO(weights_path)
        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        model.predict(source=dummy, imgsz=64, device="cpu", verbose=False)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_ssd_model(weights_path: str):
    """
    Loads and caches the trained SSDLite320 MobileNetV3 model without external downloads.
    Runs warmup inference to pre-allocate PyTorch CPU buffers.
    """
    try:
        torch.set_num_threads(4)
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        in_channels = [672, 480, 512, 256, 256, 128]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=2,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Warmup
        dummy = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            _ = model(dummy)
        return model
    except Exception as e:
        st.error(f"Error loading SSD model: {e}")
        return None

# -----------------------------------------------------------------------------
# Inference Helpers
# -----------------------------------------------------------------------------
def run_yolo_inference(model, image: Image.Image, conf_threshold: float, iou_threshold: float):
    """
    Runs YOLO prediction with timing and returns list of detections.
    """
    t0 = time.time()
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=416,
        device="cpu",
        verbose=False
    )
    t_inference = (time.time() - t0) * 1000  # ms
    
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes_data = results[0].boxes
        for i in range(len(boxes_data)):
            xyxy = boxes_data.xyxy[i].cpu().numpy()
            conf_val = float(boxes_data.conf[i].cpu().numpy())
            cls_idx = int(boxes_data.cls[i].cpu().numpy())
            detections.append({
                "id": i + 1,
                "box": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                "confidence": conf_val,
                "class_name": model.names.get(cls_idx, "Pothole")
            })
    return detections, t_inference

def run_ssd_inference(model, image: Image.Image, conf_threshold: float):
    """
    Preprocesses to 320x320, runs SSD inference on CPU, and scales back coordinates.
    """
    orig_w, orig_h = image.size
    resized = image.resize((320, 320), Image.BILINEAR)
    img_tensor = TF.to_tensor(resized)
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model([img_tensor])
    t_inference = (time.time() - t0) * 1000  # ms
    
    out = outputs[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    
    scale_x = orig_w / 320.0
    scale_y = orig_h / 320.0
    
    detections = []
    det_id = 1
    for i in range(len(scores)):
        if labels[i] == 1 and scores[i] >= conf_threshold:
            b = boxes[i]
            scaled_box = [
                float(b[0] * scale_x),
                float(b[1] * scale_y),
                float(b[2] * scale_x),
                float(b[3] * scale_y)
            ]
            detections.append({
                "id": det_id,
                "box": scaled_box,
                "confidence": float(scores[i]),
                "class_name": "Pothole"
            })
            det_id += 1
            
    return detections, t_inference

# -----------------------------------------------------------------------------
# Custom Bounding Box Drawing Function (with dynamic label sizing & color theme)
# -----------------------------------------------------------------------------
def draw_detections(image: Image.Image, detections: list, primary_color="#2563eb", label_bg="#1d4ed8", line_width: int = 3) -> Image.Image:
    """
    Draws bounding boxes and confidence badges. Supports custom theme color.
    Uses compact labels for small images (width < 400px) to prevent overlap.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    img_w = annotated.width
    is_small = img_w < 400

    if is_small:
        font_size = max(9, int(img_w * 0.028))
    else:
        font_size = max(13, int(img_w * 0.022))

    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()
        
    text_color = "#ffffff"
    effective_line_width = max(1, line_width - 1) if is_small else line_width

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["confidence"]

        label = f"{conf:.0%}" if is_small else f"Pothole {conf:.0%}"
        
        draw.rectangle([x1, y1, x2, y2], outline=primary_color, width=effective_line_width)
        
        try:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = (40 if is_small else 80, 12 if is_small else 16)

        pad_x = 4 if is_small else 10
        pad_y = 2 if is_small else 6
            
        badge_y1 = max(0, y1 - text_h - pad_y)
        badge_y2 = badge_y1 + text_h + pad_y
        badge_x1 = x1
        badge_x2 = x1 + text_w + pad_x

        if badge_x2 > img_w:
            badge_x1 = max(0, img_w - text_w - pad_x)
            badge_x2 = img_w
        
        draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=label_bg)
        draw.text((badge_x1 + pad_x // 2, badge_y1 + 1), label, fill=text_color, font=font)
        
    return annotated

# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def render_vehicle_detection_ui():
    st.markdown('<div class="app-subtitle">Multi-Class Vehicle Detection: Ambulance, Bus, Car, Motorcycle, Truck</div>', unsafe_allow_html=True)
    
    VEHICLE_MODEL_PATH = BASE_DIR / "models" / "vehicle_best.pt"
    VEHICLE_FALLBACK_PATH = BASE_DIR / "runs" / "detect" / "train_vehicle" / "weights" / "best.pt"
    
    active_vehicle_path = VEHICLE_MODEL_PATH if VEHICLE_MODEL_PATH.exists() else (VEHICLE_FALLBACK_PATH if VEHICLE_FALLBACK_PATH.exists() else None)
    
    if not active_vehicle_path:
        st.markdown(f"""
        <div class="upload-empty">
            <div class="upload-icon">🚗</div>
            <div class="upload-title">Model Training Required</div>
            <div class="upload-sub">Missing model: `{VEHICLE_MODEL_PATH}`<br><br>Please run the vehicle training script to generate the model weights.</div>
        </div>
        """, unsafe_allow_html=True)
        render_footer()
        return
        
    vehicle_model = load_yolo_model(str(active_vehicle_path))
    
    with st.sidebar:
        st.markdown("### Detection Thresholds")
        conf_threshold = st.slider("Confidence", 0.05, 1.00, 0.25, 0.05, key="vehicle_conf")
        iou_threshold = st.slider("IoU (NMS)", 0.20, 0.80, 0.45, 0.05, key="vehicle_iou")
    
    st.markdown('<div class="section-title">1. Upload Image</div>', unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # MODIFIED: ACCEPT MULTIPLE FILES FOR COLLAGE
    # -------------------------------------------------------------------------
    uploaded_files = st.file_uploader("Upload images (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed", accept_multiple_files=True)
    
    if not uploaded_files:
        st.markdown("""
        <div class="upload-empty">
            <div class="upload-icon">📸</div>
            <div class="upload-title">Upload an image to detect vehicles</div>
            <div class="upload-sub">Drag and drop or click to browse files (JPG, PNG)</div>
        </div>
        """, unsafe_allow_html=True)
        render_footer()
        return

    st.markdown('<div class="section-title">2. Visual Results (Collage)</div>', unsafe_allow_html=True)
    
    num_images = len(uploaded_files)
    num_cols = min(num_images, 3)
    cols = st.columns(num_cols)
    
    color_map = {
        "Ambulance": "#ef4444",
        "Bus": "#f59e0b",
        "Car": "#3b82f6",
        "Motorcycle": "#10b981",
        "Truck": "#8b5cf6"
    }
    
    for idx, uploaded_file in enumerate(uploaded_files):
        col = cols[idx % num_cols]
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Unable to read uploaded file: {e}")
            continue
            
        with col:
            dets, latency = run_yolo_inference(vehicle_model, input_image, conf_threshold, iou_threshold)
            
            annotated_img = input_image.copy()
            draw = ImageDraw.Draw(annotated_img)
            try:
                font = ImageFont.truetype("arial.ttf", size=max(12, int(annotated_img.width * 0.025)))
            except:
                font = ImageFont.load_default()
                
            for det in dets:
                cls_name = det["class_name"]
                box_color = color_map.get(cls_name, "#64748b")
                x1, y1, x2, y2 = det["box"]
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                
                label = f"{cls_name} {det['confidence']:.0%}"
                try:
                    bbox = font.getbbox(label)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w, text_h = (80, 15)
                    
                badge_y1 = max(0, y1 - text_h - 4)
                draw.rectangle([x1, badge_y1, x1 + text_w + 6, badge_y1 + text_h + 4], fill=box_color)
                draw.text((x1 + 3, badge_y1 + 1), label, fill="white", font=font)
                
            st.image(annotated_img, caption=f"{uploaded_file.name} | {len(dets)} det ({latency:.0f}ms)", use_container_width=True)
            
    render_footer()

def main():
    st.markdown('<div class="app-title">Computer Vision AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Real-time object detection and visual intelligence</div>', 
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.markdown("### Navigation")
        project_scope = st.radio("Select Project", ["Pothole Detection", "Vehicle Detection"], label_visibility="collapsed")
        st.markdown("---")

    if project_scope == "Vehicle Detection":
        st.markdown('<div class="status-badge"><span class="status-dot"></span>Vehicle Detection System Active</div>', unsafe_allow_html=True)
        render_vehicle_detection_ui()
        return

    st.markdown('<div class="status-badge"><span class="status-dot"></span>Pothole Detection System Active</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. MODE SELECTOR
    # -------------------------------------------------------------------------
    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #334155; margin-bottom: 0.35rem;'>Select Detection Mode:</p>", unsafe_allow_html=True)
    
    selected_mode = st.radio(
        "Detection Mode",
        options=["YOLO11n", "SSD", "Compare Both"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    # Resolve Model Weights
    active_yolo_path = YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else (YOLO_FALLBACK_PATH if YOLO_FALLBACK_PATH.exists() else None)
    active_ssd_path = SSD_MODEL_PATH if SSD_MODEL_PATH.exists() else (SSD_FALLBACK_PATH if SSD_FALLBACK_PATH.exists() else None)

    # Status Badge
    if selected_mode == "YOLO11n":
        if active_yolo_path:
            yolo_model = load_yolo_model(str(active_yolo_path))
            st.markdown('<div class="status-badge"><span class="status-dot"></span>YOLO11n Ready (5.18 MB)</div>', unsafe_allow_html=True)
        else:
            st.error("Missing YOLO model weights at `models/yolo_best.pt`.")
            st.stop()
            return
            
    elif selected_mode == "SSD":
        if active_ssd_path:
            ssd_model = load_ssd_model(str(active_ssd_path))
            st.markdown('<div class="status-badge"><span class="status-dot"></span>SSDLite320 MobileNetV3 Ready (8.70 MB)</div>', unsafe_allow_html=True)
        else:
            st.error("Missing SSD model weights at `models/ssd_best.pth`.")
            st.stop()
            return
            
    else:  # Compare Both
        if active_yolo_path and active_ssd_path:
            yolo_model = load_yolo_model(str(active_yolo_path))
            ssd_model = load_ssd_model(str(active_ssd_path))
            st.markdown('<div class="status-badge"><span class="status-dot"></span>Both Models Ready (YOLO11n & SSDLite320 MobileNetV3)</div>', unsafe_allow_html=True)
        else:
            st.error("Missing model weights for comparison.")
            st.stop()
            return

    # -------------------------------------------------------------------------
    # 3. SIDEBAR CONFIGURATION & BENCHMARK METRICS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=1.00,
            value=0.25,
            step=0.05,
            help="Minimum prediction confidence required to count a detection."
        )

        if selected_mode in ["YOLO11n", "Compare Both"]:
            iou_threshold = st.slider(
                "IoU / NMS Threshold (YOLO)",
                min_value=0.20,
                max_value=0.80,
                value=0.45,
                step=0.05,
                help="Controls overlap suppression for YOLO bounding boxes."
            )
        else:
            iou_threshold = 0.45
            
        box_thickness = st.slider(
            "Bounding Box Width",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            help="Thickness of bounding boxes in pixels."
        )

        st.markdown("---")

        # Sidebar benchmark metrics based on mode
        if selected_mode == "YOLO11n":
            st.markdown("### Model Information")
            st.markdown(f"""
            - **Architecture:** `YOLO11n`
            - **Input Resolution:** `416 × 416`
            - **Model Size:** `{YOLO_METRICS['model_size']}`
            - **Inference Device:** `CPU`
            - **Weights File:** `{active_yolo_path.name}`
            """)
            st.markdown("---")
            st.markdown("### Test Set Performance")
            st.caption("Evaluated on 133-image Test Split (299 potholes):")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precision", YOLO_METRICS["precision"])
                st.metric("mAP@50", YOLO_METRICS["map50"])
                st.metric("F1-Score", YOLO_METRICS["f1_score"])
            with col2:
                st.metric("Recall", YOLO_METRICS["recall"])
                st.metric("mAP@50-95", YOLO_METRICS["map50_95"])
                st.metric("CPU Latency", YOLO_METRICS["inference_speed"])

        elif selected_mode == "SSD":
            st.markdown("### Model Information")
            st.markdown(f"""
            - **Architecture:** `SSDLite320 MobileNetV3-Large`
            - **Input Resolution:** `320 × 320`
            - **Model Size:** `{SSD_METRICS['model_size']}`
            - **Inference Device:** `CPU`
            - **Weights File:** `{active_ssd_path.name}`
            """)
            st.markdown("---")
            st.markdown("### Test Set Performance")
            st.caption("Evaluated on 133-image Test Split (299 potholes):")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precision", SSD_METRICS["precision"])
                st.metric("mAP@50", SSD_METRICS["map50"])
                st.metric("F1-Score", SSD_METRICS["f1_score"])
            with col2:
                st.metric("Recall", SSD_METRICS["recall"])
                st.metric("mAP@50-95", SSD_METRICS["map50_95"])
                st.metric("CPU Latency", SSD_METRICS["inference_speed"])

        else:  # Compare Both
            st.markdown("### Comparison Overview")
            st.markdown("""
            Both detectors evaluated on identical 133-image Test Split (299 ground-truth potholes).
            """)
            st.markdown(f"""
            - **YOLO11n mAP@50:** `{YOLO_METRICS['map50']}`
            - **SSD mAP@50:** `{SSD_METRICS['map50']}`
            - **YOLO11n Latency:** `{YOLO_METRICS['inference_speed']}`
            - **SSD Latency:** `{SSD_METRICS['inference_speed']}`
            """)
            st.info("YOLO11n delivers higher accuracy, recall, and faster CPU inference.")

    # -------------------------------------------------------------------------
    # 4. IMAGE SELECTION & UPLOAD AREA
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">1. Select or Upload Road Image</div>', unsafe_allow_html=True)
    
    sample_images = []
    if TEST_IMAGES_DIR.exists():
        sample_images = sorted([f.name for f in TEST_IMAGES_DIR.glob("*.jpg")])[:8]
    
    input_col1, input_col2 = st.columns([1.2, 1])
    
    with input_col1:
        uploaded_file = st.file_uploader(
            "Upload a road image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select an image from your computer to detect potholes."
        )
        
    with input_col2:
        selected_sample = None
        if sample_images:
            st.markdown("<p style='font-size: 0.9rem; font-weight: 500; color: #475569; margin-bottom: 0.35rem;'>Or choose a sample test image:</p>", unsafe_allow_html=True)
            sample_options = ["None"] + sample_images
            chosen_sample = st.selectbox(
                "Choose test sample",
                options=sample_options,
                index=0,
                label_visibility="collapsed"
            )
            if chosen_sample != "None":
                selected_sample = TEST_IMAGES_DIR / chosen_sample

    input_image = None
    image_source_label = ""
    
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            image_source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Unable to read uploaded file: {e}")
            input_image = None
    elif selected_sample is not None and selected_sample.exists():
        try:
            input_image = Image.open(selected_sample).convert("RGB")
            image_source_label = f"Test Sample: {selected_sample.name}"
        except Exception as e:
            st.error(f"Unable to load sample image: {e}")
            input_image = None

    # -------------------------------------------------------------------------
    # 5. EMPTY STATE
    # -------------------------------------------------------------------------
    if input_image is None:
        st.markdown("---")
        mode_desc = (
            "The YOLO11n model will localize and count potholes in real time." if selected_mode == "YOLO11n"
            else "The SSDLite320 MobileNetV3 detector will localize potholes in real time." if selected_mode == "SSD"
            else "Both YOLO11n and SSDLite320 models will execute on the same image for side-by-side comparison."
        )
        st.markdown(f"""
        <div style="background-color: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 8px; padding: 2.5rem 1.5rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🛣️</div>
            <p style="font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">
                Upload a road image to detect potholes
            </p>
            <p style="font-size: 0.9rem; color: #64748b; margin: 0 auto; max-width: 520px;">
                Choose an image from your device or select one of the verified test samples above. {mode_desc}
            </p>
        </div>
        """, unsafe_allow_html=True)
        render_footer()
        return

    img_w, img_h = input_image.size

    # -------------------------------------------------------------------------
    # 6. INFERENCE EXECUTION BY MODE
    # -------------------------------------------------------------------------
    if selected_mode == "YOLO11n":
        yolo_dets, yolo_latency = run_yolo_inference(yolo_model, input_image, conf_threshold, iou_threshold)
        annotated_img = draw_detections(input_image, yolo_dets, primary_color="#2563eb", label_bg="#1d4ed8", line_width=box_thickness)
        
        # Summary Cards
        st.markdown("---")
        st.markdown('<div class="section-title">2. Detection Summary (YOLO11n)</div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        num_det = len(yolo_dets)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{num_det}</div><div class="metric-label">Potholes Detected</div></div>', unsafe_allow_html=True)
        with c2:
            highest_conf = f"{max([d['confidence'] for d in yolo_dets]):.1%}" if num_det > 0 else "N/A"
            st.markdown(f'<div class="metric-box"><div class="metric-value">{highest_conf}</div><div class="metric-label">Highest Confidence</div></div>', unsafe_allow_html=True)
        with c3:
            avg_conf = f"{np.mean([d['confidence'] for d in yolo_dets]):.1%}" if num_det > 0 else "N/A"
            st.markdown(f'<div class="metric-box"><div class="metric-value">{avg_conf}</div><div class="metric-label">Average Confidence</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{yolo_latency:.0f} ms</div><div class="metric-label">Inference Latency</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-title">3. Visual Comparison</div>', unsafe_allow_html=True)
        col_orig, col_pred = st.columns(2)
        with col_orig:
            st.markdown("<p style='font-weight: 600; color: #334155; margin-bottom: 0.25rem;'>Original Image</p>", unsafe_allow_html=True)
            st.image(input_image, width="stretch")
            st.caption(f"{image_source_label} | Dimensions: {img_w} × {img_h} px")
        with col_pred:
            st.markdown("<p style='font-weight: 600; color: #1e3a8a; margin-bottom: 0.25rem;'>YOLO11n Detection</p>", unsafe_allow_html=True)
            st.image(annotated_img, width="stretch")
            st.caption(f"Detected {num_det} pothole(s) at confidence ≥ {conf_threshold:.2f} | IoU: {iou_threshold:.2f}")

        render_detection_table(yolo_dets)

    elif selected_mode == "SSD":
        ssd_dets, ssd_latency = run_ssd_inference(ssd_model, input_image, conf_threshold)
        annotated_img = draw_detections(input_image, ssd_dets, primary_color="#e11d48", label_bg="#be123c", line_width=box_thickness)
        
        # Summary Cards
        st.markdown("---")
        st.markdown('<div class="section-title">2. Detection Summary (SSDLite320 MobileNetV3)</div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        num_det = len(ssd_dets)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{num_det}</div><div class="metric-label">Potholes Detected</div></div>', unsafe_allow_html=True)
        with c2:
            highest_conf = f"{max([d['confidence'] for d in ssd_dets]):.1%}" if num_det > 0 else "N/A"
            st.markdown(f'<div class="metric-box"><div class="metric-value">{highest_conf}</div><div class="metric-label">Highest Confidence</div></div>', unsafe_allow_html=True)
        with c3:
            avg_conf = f"{np.mean([d['confidence'] for d in ssd_dets]):.1%}" if num_det > 0 else "N/A"
            st.markdown(f'<div class="metric-box"><div class="metric-value">{avg_conf}</div><div class="metric-label">Average Confidence</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{ssd_latency:.0f} ms</div><div class="metric-label">Inference Latency</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-title">3. Visual Comparison</div>', unsafe_allow_html=True)
        col_orig, col_pred = st.columns(2)
        with col_orig:
            st.markdown("<p style='font-weight: 600; color: #334155; margin-bottom: 0.25rem;'>Original Image</p>", unsafe_allow_html=True)
            st.image(input_image, width="stretch")
            st.caption(f"{image_source_label} | Dimensions: {img_w} × {img_h} px")
        with col_pred:
            st.markdown("<p style='font-weight: 600; color: #be123c; margin-bottom: 0.25rem;'>SSD Detection</p>", unsafe_allow_html=True)
            st.image(annotated_img, width="stretch")
            st.caption(f"Detected {num_det} pothole(s) at confidence ≥ {conf_threshold:.2f}")

        render_detection_table(ssd_dets)

    else:  # Compare Both Mode
        yolo_dets, yolo_latency = run_yolo_inference(yolo_model, input_image, conf_threshold, iou_threshold)
        ssd_dets, ssd_latency = run_ssd_inference(ssd_model, input_image, conf_threshold)
        
        yolo_annotated = draw_detections(input_image, yolo_dets, primary_color="#2563eb", label_bg="#1d4ed8", line_width=box_thickness)
        ssd_annotated = draw_detections(input_image, ssd_dets, primary_color="#e11d48", label_bg="#be123c", line_width=box_thickness)
        
        st.markdown("---")
        st.markdown('<div class="section-title">2. Side-by-Side Visual Comparison</div>', unsafe_allow_html=True)
        
        col_yolo_view, col_ssd_view = st.columns(2)
        
        with col_yolo_view:
            st.markdown(
                '<span class="model-tag-yolo">YOLO11n (Pretrained)</span> '
                f'<strong>{len(yolo_dets)} Pothole(s)</strong> &bull; Latency: <strong>{yolo_latency:.0f} ms</strong>', 
                unsafe_allow_html=True
            )
            st.image(yolo_annotated, width="stretch")
            st.caption(f"YOLO11n: {len(yolo_dets)} detections at conf ≥ {conf_threshold:.2f}, IoU={iou_threshold:.2f}")

        with col_ssd_view:
            st.markdown(
                '<span class="model-tag-ssd">SSDLite320 MobileNetV3</span> '
                f'<strong>{len(ssd_dets)} Pothole(s)</strong> &bull; Latency: <strong>{ssd_latency:.0f} ms</strong>', 
                unsafe_allow_html=True
            )
            st.image(ssd_annotated, width="stretch")
            st.caption(f"SSDLite320: {len(ssd_dets)} detections at conf ≥ {conf_threshold:.2f}")

        # Current Run Metrics Comparison
        st.markdown('<div class="section-title">3. Current Run Comparison</div>', unsafe_allow_html=True)
        
        c_y1, c_y2, c_s1, c_s2 = st.columns(4)
        with c_y1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{len(yolo_dets)}</div><div class="metric-label">YOLO11n Detections</div></div>', unsafe_allow_html=True)
        with c_y2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{yolo_latency:.0f} ms</div><div class="metric-label">YOLO11n Latency</div></div>', unsafe_allow_html=True)
        with c_s1:
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color: #be123c;">{len(ssd_dets)}</div><div class="metric-label">SSD Detections</div></div>', unsafe_allow_html=True)
        with c_s2:
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color: #be123c;">{ssd_latency:.0f} ms</div><div class="metric-label">SSD Latency</div></div>', unsafe_allow_html=True)

        # Benchmark Comparison Table
        st.markdown('<div class="section-title">4. Comprehensive Test Set Benchmark Comparison</div>', unsafe_allow_html=True)
        st.caption("Rigorous evaluation on the identical 133-image independent Test Split (299 ground-truth potholes):")

        comparison_data = [
            {"Metric": "Precision (IoU 0.50)", "YOLO11n": YOLO_METRICS["precision"], "SSDLite MobileNetV3": SSD_METRICS["precision"], "Advantage": "YOLO (+53.56%)"},
            {"Metric": "Recall (IoU 0.50)", "YOLO11n": YOLO_METRICS["recall"], "SSDLite MobileNetV3": SSD_METRICS["recall"], "Advantage": "YOLO (+16.39%)"},
            {"Metric": "F1-Score", "YOLO11n": YOLO_METRICS["f1_score"], "SSDLite MobileNetV3": SSD_METRICS["f1_score"], "Advantage": "YOLO (+39.43%)"},
            {"Metric": "mAP@50", "YOLO11n": YOLO_METRICS["map50"], "SSDLite MobileNetV3": SSD_METRICS["map50"], "Advantage": "YOLO (+38.02%)"},
            {"Metric": "mAP@50-95", "YOLO11n": YOLO_METRICS["map50_95"], "SSDLite MobileNetV3": SSD_METRICS["map50_95"], "Advantage": "YOLO (+29.94%)"},
            {"Metric": "CPU Inference Latency", "YOLO11n": YOLO_METRICS["inference_speed"], "SSDLite MobileNetV3": SSD_METRICS["inference_speed"], "Advantage": "YOLO is ~25 ms faster"},
            {"Metric": "Throughput (FPS)", "YOLO11n": YOLO_METRICS["fps"], "SSDLite MobileNetV3": SSD_METRICS["fps"], "Advantage": "YOLO is ~1.3 FPS faster"},
            {"Metric": "Model File Size", "YOLO11n": YOLO_METRICS["model_size"], "SSDLite MobileNetV3": SSD_METRICS["model_size"], "Advantage": "YOLO is 40% more compact"},
            {"Metric": "Training Time (10 ep)", "YOLO11n": YOLO_METRICS["training_time"], "SSDLite MobileNetV3": SSD_METRICS["training_time"], "Advantage": "Comparable on CPU"}
        ]
        
        st.dataframe(comparison_data, width="stretch", hide_index=True)

        # Highlight Box
        st.markdown("""
        <div class="winner-box">
            <h4 style="color: #166534; margin: 0 0 0.35rem 0;">🏆 Benchmark Winner: YOLO11n</h4>
            <p style="color: #14532d; margin: 0; font-size: 0.92rem; line-height: 1.5;">
                <strong>YOLO11n significantly outperforms SSDLite320 MobileNetV3</strong> across every quantitative metric:
                achieving <strong>70.94% mAP@50</strong> vs. 32.92% for SSD (+38.02% gain), 
                <strong>74.03% precision</strong> vs. 20.47% (+53.56% gain), and lower inference latency (<strong>~101 ms</strong> vs. 126 ms on CPU) 
                while maintaining a 40% smaller model footprint (5.18 MB vs. 8.70 MB).
            </p>
        </div>
        """, unsafe_allow_html=True)

    render_footer()

def render_detection_table(detections):
    if len(detections) == 0:
        st.info("ℹ️ **No pothole detected** above the selected confidence threshold. Try adjusting the slider in the sidebar.")
    else:
        with st.expander(f"📋 Detailed Detection Coordinates ({len(detections)} Pothole{'s' if len(detections) > 1 else ''})", expanded=False):
            table_rows = []
            for d in detections:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                table_rows.append({
                    "Detection ID": f"Pothole #{d['id']}",
                    "Confidence": f"{d['confidence']:.2%}",
                    "Bounding Box (x1, y1, x2, y2)": f"[{x1}, {y1}, {x2}, {y2}]",
                    "Width (px)": x2 - x1,
                    "Height (px)": y2 - y1,
                    "Area (px²)": (x2 - x1) * (y2 - y1)
                })
            st.dataframe(table_rows, width="stretch", hide_index=True)

def render_footer():
    st.markdown("""
    <div class="app-footer">
        Computer Vision Project &bull; Pothole Detection & Model Benchmarking (YOLO11n vs. SSDLite320 MobileNetV3)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
