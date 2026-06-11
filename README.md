# 🏠 HouseHub
**Anti-Fraud Public Housing Allocation System**

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Built with ❤️ by TEAM HouseHub**

HouseHub is a full-stack, AI-powered public housing allocation platform designed to ensure fair, transparent, and fraud-resistant distribution of housing resources. By combining deterministic Machine Learning scoring with Computer Vision forgery detection, HouseHub automates the allocation process while strictly preserving citizen privacy.

---

## ✨ Key Features

1. **🔒 Privacy-First Design (`PrivacyShield`)**
   - All Personally Identifiable Information (PII) such as Names and National IDs are deterministically hashed using SHA-256 immediately upon ingestion.
   - Zero raw PII is stored in the database or session state.

2. **🕵️ AI Forgery Detection (`DocumentInspector`)**
   - Employs Computer Vision (OpenCV) to simulate Error Level Analysis (ELA).
   - Detects digital manipulation and non-uniform recompression artifacts in uploaded ID and income documents.

3. **🤖 Objective ML Scoring (`RankingEngine`)**
   - Utilizes a Scikit-Learn `RandomForestRegressor` trained on synthetic datasets reflecting real-world housing need patterns (rent burden, dependents, current housing quality).
   - Generates a transparent, auditable "Need Score" (0–100) to prioritize applicants.

4. **⚖️ Dual-Sided UI**
   - **Citizen Portal:** A clean, accessible interface for citizens to securely upload their documents and demographic data.
   - **Admin Dashboard:** A control center featuring a live demand heatmap, a sorted "Clean Queue" for verified applicants, and a highlighted "Red Alert" queue for manual investigation of flagged documents.

---

## 🛠️ Architecture & Tech Stack

- **Frontend / Routing:** Streamlit
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Computer Vision:** OpenCV (`opencv-python-headless`), Pillow
- **Geospatial Mapping:** Folium, `streamlit-folium`
- **OCR (Simulation):** EasyOCR (Ready for production text-extraction)

---

## 🚀 Getting Started

### Prerequisites

You must have [Conda](https://docs.conda.io/en/latest/) installed on your system.

### 1. Environment Setup

It is highly recommended to use a dedicated Conda environment to manage dependencies:

```bash
# Create the environment
conda create -n househub_env python=3.11 -y

# Activate the environment
conda activate househub_env
```

### 2. Install Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

*(Note: Depending on your system, you may need to install the CPU-only version of PyTorch for EasyOCR to keep the installation lightweight).*

### 3. Run the Application

Launch the Streamlit server:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 🏗️ Project Structure

```text
househub/
│
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
└── modules/
    ├── __init__.py             # Package initializer
    ├── privacy_shield.py       # PII anonymization & hashing logic
    ├── vision_ocr.py           # CV forgery detection and OCR simulation
    └── ml_scoring.py           # ML Need Score calculation engine
```

---

## 👥 Acknowledgements

Designed and developed by **TEAM HouseHub**. Dedicated to building ethical AI tools for the public sector. 
