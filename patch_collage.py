import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# -------------------------------------------------------------
# 1. Patch Vehicle Detection
# -------------------------------------------------------------
vehicle_uploader_old = """    st.markdown('<div class="section-title">1. Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is None:
        st.markdown(\"\"\"
        <div class="upload-empty">
            <div class="upload-icon">📸</div>
            <div class="upload-title">Upload an image to detect vehicles</div>
            <div class="upload-sub">Drag and drop or click to browse files (JPG, PNG)</div>
        </div>
        \"\"\", unsafe_allow_html=True)
        render_footer()
        return

    try:
        input_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Unable to read uploaded file: {e}")
        render_footer()
        return
        
    dets, latency = run_yolo_inference(vehicle_model, input_image, conf_threshold, iou_threshold)
    
    color_map = {
        "Ambulance": "#ef4444",
        "Bus": "#f59e0b",
        "Car": "#3b82f6",
        "Motorcycle": "#10b981",
        "Truck": "#8b5cf6"
    }
    
    annotated_img = input_image.copy()
    draw = ImageDraw.Draw(annotated_img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, int(annotated_img.width * 0.025)))
    except:
        font = ImageFont.load_default()
        
    counts = {"Ambulance": 0, "Bus": 0, "Car": 0, "Motorcycle": 0, "Truck": 0}
    confs = []
    
    for det in dets:
        cls_name = det["class_name"]
        if cls_name in counts:
            counts[cls_name] += 1
        confs.append(det["confidence"])
        
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
        
    num_det = len(dets)
    avg_conf = np.mean(confs) if num_det > 0 else 0
    max_conf = np.max(confs) if num_det > 0 else 0
    
    st.markdown('<div class="section-title">2. Detection Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{num_det}</div><div class="metric-label">Objects Detected</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{max_conf:.1%}</div><div class="metric-label">Highest Confidence</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{avg_conf:.1%}</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{latency:.0f} ms</div><div class="metric-label">Processing Time</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">3. Visual Results</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(annotated_img, use_container_width=True)
        
    with col2:
        st.markdown("#### Class Breakdown")
        for cls_name, count in counts.items():
            st.markdown(f"- **{cls_name}:** {count}")
            
    if num_det > 0:
        st.markdown('<div class="section-title">4. Detection Details</div>', unsafe_allow_html=True)
        table_data = []
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            table_data.append({
                "Class": d["class_name"],
                "Confidence": f"{d['confidence']:.2%}",
                "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]"
            })
        st.dataframe(table_data, use_container_width=True)"""

vehicle_uploader_new = """    st.markdown('<div class="section-title">1. Upload Image</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed", accept_multiple_files=True)
    
    if not uploaded_files:
        st.markdown(\"\"\"
        <div class="upload-empty">
            <div class="upload-icon">📸</div>
            <div class="upload-title">Upload an image to detect vehicles</div>
            <div class="upload-sub">Drag and drop or click to browse files (JPG, PNG)</div>
        </div>
        \"\"\", unsafe_allow_html=True)
        render_footer()
        return

    st.markdown('<div class="section-title">2. Visual Results (Collage)</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    
    color_map = {
        "Ambulance": "#ef4444",
        "Bus": "#f59e0b",
        "Car": "#3b82f6",
        "Motorcycle": "#10b981",
        "Truck": "#8b5cf6"
    }
    
    for idx, uploaded_file in enumerate(uploaded_files):
        col = cols[idx % 3]
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
                
            st.image(annotated_img, caption=f"{uploaded_file.name} | {len(dets)} det ({latency:.0f}ms)", use_container_width=True)"""

content = content.replace(vehicle_uploader_old, vehicle_uploader_new)

# -------------------------------------------------------------
# 2. Patch Pothole Detection Data loading
# -------------------------------------------------------------
pothole_upload_old = """    st.markdown('<div class="section-title">1. Select or Upload Road Image</div>', unsafe_allow_html=True)
    
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
    if input_image is None:"""

pothole_upload_new = """    st.markdown('<div class="section-title">1. Select or Upload Road Images</div>', unsafe_allow_html=True)
    
    sample_images = []
    if TEST_IMAGES_DIR.exists():
        sample_images = sorted([f.name for f in TEST_IMAGES_DIR.glob("*.jpg")])[:8]
    
    input_col1, input_col2 = st.columns([1.2, 1])
    
    with input_col1:
        uploaded_files = st.file_uploader(
            "Upload road images (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select images from your computer to detect potholes.",
            accept_multiple_files=True
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

    input_images = []
    
    if uploaded_files:
        for uf in uploaded_files:
            try:
                img = Image.open(uf).convert("RGB")
                input_images.append({"image": img, "name": uf.name})
            except Exception as e:
                st.error(f"Unable to read uploaded file {uf.name}: {e}")
    elif selected_sample is not None and selected_sample.exists():
        try:
            img = Image.open(selected_sample).convert("RGB")
            input_images.append({"image": img, "name": selected_sample.name})
        except Exception as e:
            st.error(f"Unable to load sample image: {e}")

    # -------------------------------------------------------------------------
    # 5. EMPTY STATE
    # -------------------------------------------------------------------------
    if not input_images:"""

content = content.replace(pothole_upload_old, pothole_upload_new)


# -------------------------------------------------------------
# 3. Patch Pothole Detection Inference execution
# -------------------------------------------------------------
pothole_inference_start = """    img_w, img_h = input_image.size

    # -------------------------------------------------------------------------
    # 6. INFERENCE EXECUTION BY MODE
    # -------------------------------------------------------------------------"""

pothole_inference_end = """    render_footer()

def render_detection_table"""

pothole_inference_new = """    # -------------------------------------------------------------------------
    # 6. INFERENCE EXECUTION BY MODE (COLLAGE)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(f'<div class="section-title">2. Visual Results ({len(input_images)} Image{"s" if len(input_images) > 1 else ""})</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    
    for idx, img_data in enumerate(input_images):
        col = cols[idx % 3]
        input_image = img_data["image"]
        image_source_label = img_data["name"]
        
        with col:
            st.markdown(f"<p style='font-weight:600;margin-bottom:0.1rem;'>{image_source_label}</p>", unsafe_allow_html=True)
            
            if selected_mode == "YOLO11n":
                yolo_dets, yolo_latency = run_yolo_inference(yolo_model, input_image, conf_threshold, iou_threshold)
                annotated_img = draw_detections(input_image, yolo_dets, primary_color="#2563eb", label_bg="#1d4ed8", line_width=box_thickness)
                st.image(annotated_img, use_container_width=True)
                st.caption(f"YOLO11n: {len(yolo_dets)} det | {yolo_latency:.0f} ms")
                
            elif selected_mode == "SSD":
                ssd_dets, ssd_latency = run_ssd_inference(ssd_model, input_image, conf_threshold)
                annotated_img = draw_detections(input_image, ssd_dets, primary_color="#e11d48", label_bg="#be123c", line_width=box_thickness)
                st.image(annotated_img, use_container_width=True)
                st.caption(f"SSD: {len(ssd_dets)} det | {ssd_latency:.0f} ms")
                
            else:  # Compare Both
                yolo_dets, yolo_latency = run_yolo_inference(yolo_model, input_image, conf_threshold, iou_threshold)
                ssd_dets, ssd_latency = run_ssd_inference(ssd_model, input_image, conf_threshold)
                
                yolo_annotated = draw_detections(input_image, yolo_dets, primary_color="#2563eb", label_bg="#1d4ed8", line_width=box_thickness)
                ssd_annotated = draw_detections(input_image, ssd_dets, primary_color="#e11d48", label_bg="#be123c", line_width=box_thickness)
                
                st.image(yolo_annotated, use_container_width=True)
                st.caption(f"YOLO11n: {len(yolo_dets)} det | {yolo_latency:.0f} ms")
                st.image(ssd_annotated, use_container_width=True)
                st.caption(f"SSD: {len(ssd_dets)} det | {ssd_latency:.0f} ms")
"""

start_idx = content.find(pothole_inference_start)
end_idx = content.find(pothole_inference_end)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + pothole_inference_new + "\n" + content[end_idx:]

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Collage Patch Applied Successfully.")
