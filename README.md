# 🛍️ Cloud-Based Customer Segmentation for E-Commerce Using RFM Analysis and K-Means Clustering

**College Project Prototype — Local Execution & Synthetic Dataset Foundation**

---

## 📌 Project Overview

This application is a data science & machine learning prototype designed for e-commerce customer segmentation. It processes transactional dataset records, calculates **Recency, Frequency, and Monetary (RFM)** metrics, applies **K-Means Clustering** machine learning models, and assigns actionable business persona labels to customers.

---

## ☁️ Architecture & Future Cloud Roadmap

While this initial version runs **100% locally on Windows** without external paid cloud APIs, it is architected with a production-ready cloud deployment roadmap in mind:

```
[ Local Prototype (Current Phase) ]
┌────────────────┐     ┌────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Synthetic /   │ ──> │ Data Loader &  │ ──> │  RFM + K-Means   │ ──> │ Streamlit &    │
│  CSV Dataset   │     │ Preprocessing  │     │  Scikit-Learn    │     │ Plotly UI      │
└────────────────┘     └────────────────┘     └──────────────────┘     └────────────────┘

[ Enterprise Cloud Architecture (Future Roadmap - Google Cloud Platform) ]
┌────────────────┐     ┌────────────────┐     ┌──────────────────┐     ┌────────────────┐
│ E-Commerce DB  │ ──> │ Google Cloud   │ ──> │ Google BigQuery  │ ──> │ Looker Studio  │
│ Transactions   │     │ Storage (GCS)  │     │ & BigQuery ML    │     │ Dashboard      │
└────────────────┘     └────────────────┘     └──────────────────┘     └────────────────┘
```

- **Storage:** Google Cloud Storage (GCS) raw dataset bucket.
- **Data Warehouse & ETL:** Google BigQuery SQL for aggregations and RFM views.
- **Machine Learning:** BigQuery ML (K-Means Clustering) scaled across large enterprise datasets.
- **Executive Dashboarding:** Looker Studio for executive reporting.

---

## 📁 Project Structure

```
customer-segmentation/
├── app.py                     # Streamlit Web Application Interface
├── requirements.txt           # Python Dependencies
├── README.md                  # Project Documentation & Setup Guide
├── data/                      # Local data directory for uploaded CSVs
├── src/                       # Core Data Science & ML Engine
│   ├── __init__.py            # Package Initialization
│   ├── data_loader.py         # Data Loading, Validation & Synthetic Demo Generator
│   ├── preprocessing.py       # Data Cleaning & Feature Scaling
│   ├── rfm.py                 # Recency, Frequency, Monetary Calculations
│   ├── clustering.py          # K-Means Clustering & Elbow/Silhouette Evaluation
│   ├── segmentation.py        # Business Persona Mapping
│   └── visualizations.py      # Plotly Interactive Visualizations
└── tests/                     # Unit Tests
    └── test_pipeline.py       # Pipeline Validation & Data Generator Tests
```

---

## 🛠️ Technology Stack

- **Language:** Python 3.12
- **Web Interface:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Visualization:** Plotly Express & Plotly Graph Objects
- **Testing:** Pytest

---

## 🚀 Step-by-Step Instructions (Windows PowerShell)

Follow these exact steps to set up and launch the application on **Windows PowerShell**:

### 1️⃣ Open PowerShell and Navigate to the Project Folder
```powershell
cd "c:\Users\tejas\OneDrive\Desktop\CCD Project\customer-segmentation"
```

### 2️⃣ Create and Activate a Python Virtual Environment
```powershell
# Create virtual environment named 'venv'
python -m venv venv

# Set Execution Policy (if script execution is restricted on Windows)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### 3️⃣ Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Run Unit Tests to Verify Setup
```powershell
pytest tests/
```

### 5️⃣ Launch the Streamlit Web Application
```powershell
streamlit run app.py
```

After running `streamlit run app.py`, your browser will automatically open at `http://localhost:8501`.

---

## 🧪 Synthetic Demo Data

The application includes a built-in **Synthetic Demo Dataset Generator** that creates realistic e-commerce transactions across diverse purchasing profiles (Champions, Loyalists, At-Risk Spenders, New Buyers, and Casual Buyers).

All generated synthetic records are clearly badged in the UI to distinguish demo data from real production datasets.
