import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Patch Vehicle Detection
vehicle_uploader_old = """    st.markdown('<div class="section-title">1. Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is None:"""

vehicle_uploader_new = """    st.markdown('<div class="section-title">1. Upload Image</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload images (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed", accept_multiple_files=True)
    
    if not uploaded_files:"""

content = content.replace(vehicle_uploader_old, vehicle_uploader_new)

# Indent vehicle inference logic
vehicle_inference_start = """    try:
        input_image = Image.open(uploaded_file).convert("RGB")"""
vehicle_inference_end = """        st.dataframe(table_data, use_container_width=True)"""

start_idx = content.find(vehicle_inference_start)
end_idx = content.find(vehicle_inference_end) + len(vehicle_inference_end)

if start_idx != -1 and end_idx != -1:
    block = content[start_idx:end_idx]
    indented_block = "\n".join(["    " + line if line.strip() else line for line in block.split("\n")])
    # Replace the try block to iterate over uploaded_files
    indented_block = indented_block.replace("        input_image = Image.open(uploaded_file).convert(\"RGB\")", "        input_image = Image.open(uploaded_file).convert(\"RGB\")")
    wrapper = "    for uploaded_file in uploaded_files:\n        st.markdown(f\"### Results for {uploaded_file.name}\")\n" + indented_block
    content = content[:start_idx] + wrapper + content[end_idx:]

# 2. Patch Pothole Detection
pothole_uploader_old = """        uploaded_file = st.file_uploader(
            "Upload a road image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select an image from your computer to detect potholes."
        )"""

pothole_uploader_new = """        uploaded_files = st.file_uploader(
            "Upload road images (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select images from your computer to detect potholes.",
            accept_multiple_files=True
        )"""

content = content.replace(pothole_uploader_old, pothole_uploader_new)

pothole_logic_old = """    input_image = None
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

pothole_logic_new = """    input_images = []
    
    if uploaded_files:
        for uf in uploaded_files:
            try:
                img = Image.open(uf).convert("RGB")
                input_images.append({"image": img, "name": f"Uploaded File: {uf.name}"})
            except Exception as e:
                st.error(f"Unable to read uploaded file {uf.name}: {e}")
    elif selected_sample is not None and selected_sample.exists():
        try:
            img = Image.open(selected_sample).convert("RGB")
            input_images.append({"image": img, "name": f"Test Sample: {selected_sample.name}"})
        except Exception as e:
            st.error(f"Unable to load sample image: {e}")

    # -------------------------------------------------------------------------
    # 5. EMPTY STATE
    # -------------------------------------------------------------------------
    if not input_images:"""

content = content.replace(pothole_logic_old, pothole_logic_new)

pothole_inference_start = """    img_w, img_h = input_image.size"""
pothole_inference_end = """        </div>
        \"\"\", unsafe_allow_html=True)"""

start_idx2 = content.find(pothole_inference_start)
end_idx2 = content.find(pothole_inference_end) + len(pothole_inference_end)

if start_idx2 != -1 and end_idx2 != -1:
    block2 = content[start_idx2:end_idx2]
    indented_block2 = "\n".join(["    " + line if line.strip() else line for line in block2.split("\n")])
    wrapper2 = "    for img_data in input_images:\n        input_image = img_data['image']\n        image_source_label = img_data['name']\n        st.markdown(f\"<h2>🖼️ Processing: {image_source_label}</h2>\", unsafe_allow_html=True)\n        img_w, img_h = input_image.size\n" + indented_block2.replace("    img_w, img_h = input_image.size\n", "", 1)
    content = content[:start_idx2] + wrapper2 + content[end_idx2:]

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied")
