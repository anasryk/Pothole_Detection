# Pothole Detection System using YOLO and SSD

A university computer vision project comparing and deploying **YOLO** and **SSD (Single Shot MultiBox Detector)** architectures for real-time pothole detection.

---

## 📌 Overview

Road conditions directly affect traffic safety and vehicle maintenance. This project aims to detect and localize potholes from road imagery and video streams using two popular object detection architectures:
1. **YOLO (You Only Look Once)**
2. **SSD (Single Shot MultiBox Detector)**

The application will feature a **Streamlit** dashboard for interactive detection and model comparison.

---

## 📂 Project Structure

```text
Pothole_Detection/
├── dataset/            # Annotated Potholes Image Dataset (from Kaggle)
├── yolo/               # YOLO training, configuration, and inference scripts
├── ssd/                # SSD model architecture, training, and evaluation scripts
├── models/             # Saved model checkpoints and weights (.pt, .pth, etc.)
├── app.py              # Main Streamlit web application entry point
├── requirements.txt    # Project dependencies and libraries
└── README.md           # Project documentation
```

---

## 📊 Dataset

- **Source:** [Annotated Potholes Image Dataset (Kaggle)](https://www.kaggle.com/)
- *Note:* The dataset will be organized into training, validation, and test splits within the `dataset/` directory.

---

## 🚀 Setup & Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd Pothole_Detection
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛣️ Next Steps

- [ ] Prepare and preprocess the Kaggle dataset
- [ ] Train and evaluate YOLO model
- [ ] Train and evaluate SSD model
- [ ] Implement Streamlit web interface for real-time detection and side-by-side comparison
