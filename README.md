# Traffic Congestion Early Warning System

> Bachelor Thesis | TU Dortmund | 2026  
> ML-based short-term traffic congestion prediction for Smart City applications

---

## Overview

This project develops and compares three machine learning models for predicting 
urban traffic congestion 20 minutes in advance, combined with a prototype 
early warning system dashboard.

**Models compared:** Random Forest · XGBoost · LSTM  
**Dataset:** Urban Traffic Speed Dataset – Guangzhou, China (Zenodo)  
**Prediction horizon:** 20 minutes  
**Application:** Smart City early warning system

---

## Results

| Model | Precision | Recall | F1 | AUC-ROC | Training Time |
|-------|-----------|--------|----|---------|---------------|
| Random Forest | 0.74 | 0.78 | 0.76 | 0.9585 | 53s |
| XGBoost | 0.67 | 0.86 | 0.75 | 0.9604 | 4.6s |
| LSTM | 0.65 | 0.87 | 0.75 | 0.9579 | 64.8s |

**Selected for deployment:** XGBoost – best balance of recall, model size (0.9 MB) and training speed.

---

## Early Warning Dashboard

The dashboard operates in two modes:

**Manual Simulation** – adjust traffic parameters and get instant predictions  
**Real Traffic Replay** – replay actual sensor data with live predictions

To run the dashboard:
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Project Structure
traffic-congestion-prediction/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_random_forest.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_lstm.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_early_warning.ipynb
├── dashboard/
│   └── app.py
├── results/
│   └── figures/
├── models/
│   ├── scaler.pkl
│   └── class_weights.pkl
├── data/
│   └── raw/
└── requirements.txt
---

## Dataset

- **Name:** Urban Traffic Speed Dataset – Guangzhou, China
- **Source:** Zenodo – DOI: [10.5281/zenodo.1205229](https://doi.org/10.5281/zenodo.1205229)
- **Size:** 1,855,572 observations
- **Roads:** 214 urban segments
- **Period:** August – September 2016
- **Interval:** 10 minutes

---

## Tech Stack
Python 3.11
scikit-learn · XGBoost · TensorFlow
Streamlit · Plotly
Pandas · NumPy · Matplotlib · Seaborn
---

## Key Findings

- All three models achieved AUC-ROC above 0.95
- XGBoost is 11x faster to train than Random Forest
- XGBoost model size (0.9 MB) is 600x smaller than Random Forest (535 MB)
- LSTM achieved lowest false negatives (5,985) – best for early warning
- Speed-based features dominate predictions across all models

---

## Author

**Abdullah Al-Qazzaz**  
B.Sc. Computer Science – TU Dortmund  
[GitHub](https://github.com/alqazzaaz) · [LinkedIn](www.linkedin.com/in/abdullah-al-qazzaz-93b316367)
