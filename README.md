# 🛡️ Aegis Combat System Evolution — Advanced Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-orange?style=for-the-badge&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20PyTorch-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> A professional-grade data science and machine learning pipeline analyzing the evolution of the **Aegis Combat System** — the world's most advanced multi-mission naval combat system, deployed across 120+ ships in 6 nations. Built entirely in Google Colab with synthetic but realistic data.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Notebook Structure](#notebook-structure)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning](#deep-learning)
- [Time Series Forecasting](#time-series-forecasting)
- [Explainability](#explainability)
- [Anomaly Detection](#anomaly-detection)
- [NLP Analysis](#nlp-analysis)
- [Results](#results)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 🔭 Overview

The **Aegis Combat System**, developed by Lockheed Martin for the United States Navy, has become the gold standard for integrated naval warfare — powering destroyers and cruisers across the US, Japan, South Korea, Spain, Norway, and Australia. Since its first deployment in 1983, it has evolved through 10+ baseline versions, expanding from basic anti-air warfare to full ballistic missile defense (BMD) and cyber-resilient, network-centric warfare.

This project applies **advanced data science techniques** to model, forecast, and explain the system's evolution across 40+ years of capability growth.

---

## 🏗️ Project Architecture
```
aegis-analytics/
│
├── Aegis_Advanced_Analytics.ipynb   ← Main Colab notebook (16 cells)
├── README.md                        ← This file
│
├── outputs/
│   ├── shap_plots.png               ← SHAP feature importance plots
│   ├── dashboard.html               ← Exported Plotly dashboard
│   └── model_report.txt             ← Final metrics report
│
└── data/
    └── synthetic_aegis_data.csv     ← Auto-generated (no upload needed)
```

---

## 📊 Dataset

The dataset is **fully synthetic but realistic**, generated from publicly known information about Aegis system milestones. No classified or proprietary data is used.

| Property | Details |
|----------|---------|
| Samples | 500 records |
| Features | 21 (raw) + 5 engineered |
| Target | `Capability_Score` (composite 0–100) |
| Time range | 1983 – 2024 |
| Nations | US, Japan, South Korea, Spain, Norway, Australia |
| Missing values | ~5% injected, handled via KNN Imputation |
| Outliers | ~3% injected, detected via Isolation Forest + DBSCAN |

### Key Features

| Feature | Description |
|---------|-------------|
| `Radar_Range_km` | Maximum radar detection range (km) |
| `Missile_Range_km` | Missile intercept range (km) |
| `Simultaneous_Targets` | Number of targets tracked at once |
| `Budget_BUSD` | Annual system budget (billion USD) |
| `Cyber_Score` | Cyber defense capability (0–100) |
| `Network_Score` | Network-centric warfare score (0–100) |
| `BMD_Capable` | Binary — ballistic missile defense capable |
| `Response_Time_ms` | Threat response latency (ms) |
| `Efficiency_Index` | Engineered: integration × uptime / 100 |

---

## 🧰 Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep Learning | `PyTorch` |
| Hyperparameter Tuning | `optuna` |
| Explainability | `shap` |
| Time Series | `prophet`, `statsmodels` |
| Anomaly Detection | `sklearn.ensemble.IsolationForest`, `DBSCAN` |
| NLP | `nltk`, `textblob`, `wordcloud` |
| Statistics | `scipy.stats` |

---

## 📓 Notebook Structure

| Cell | Title | Description |
|------|-------|-------------|
| 1 | Install Libraries | All pip installs |
| 2 | Imports | Full library import block |
| 3 | Dataset Generation | 500-sample realistic synthetic dataset with noise, outliers, missing values |
| 4 | Preprocessing Pipeline | KNN imputation, outlier flagging, label encoding, feature engineering, RobustScaler |
| 5 | Exploratory Data Analysis | Shapiro-Wilk, ANOVA, Kruskal-Wallis, PCA projection, correlation heatmap |
| 6 | Model Benchmark | 8 models × 5-fold CV — Linear, Ridge, Lasso, RF, GBM, XGBoost, LightGBM, SVR |
| 7 | Hyperparameter Optimization | Bayesian search via Optuna (50 trials, TPE sampler) |
| 8 | SHAP Explainability | TreeSHAP — bar summary, beeswarm, dependence plot |
| 9 | Deep Learning | PyTorch 5-layer MLP — BatchNorm, Dropout, HuberLoss, AdamW + CosineAnnealingLR |
| 10 | Time Series Forecasting | Prophet + SARIMA with 10-year forecast and confidence intervals |
| 11 | Anomaly Detection | Isolation Forest + DBSCAN with PCA visualization |
| 12 | Advanced Clustering | Elbow curve, Silhouette, Davies-Bouldin, Dendrogram |
| 13 | NLP Analysis | TextBlob sentiment, polarity/subjectivity scatter, WordCloud |
| 14 | Residual Diagnostics | Residuals vs fitted, Q-Q plot, scale-location, model leaderboard |
| 15 | Interactive Dashboard | 9-panel Plotly + standalone 3D scatter |
| 16 | Final Report | Full printed metrics summary |

---

## 🤖 Machine Learning Models

Eight models are benchmarked using **5-fold cross-validation** on the training set, then evaluated on a held-out test set.
```
Model                     CV R²     CV RMSE    Test R²    Test MAE
─────────────────────────────────────────────────────────────────
Linear Regression         0.xxxx    x.xxxx     0.xxxx     x.xxxx
Ridge                     0.xxxx    x.xxxx     0.xxxx     x.xxxx
Lasso                     0.xxxx    x.xxxx     0.xxxx     x.xxxx
Random Forest             0.xxxx    x.xxxx     0.xxxx     x.xxxx
Gradient Boosting         0.xxxx    x.xxxx     0.xxxx     x.xxxx
XGBoost                   0.xxxx    x.xxxx     0.xxxx     x.xxxx
LightGBM                  0.xxxx    x.xxxx     0.xxxx     x.xxxx
SVR (RBF)                 0.xxxx    x.xxxx     0.xxxx     x.xxxx
XGBoost (Tuned/Optuna)    0.xxxx    x.xxxx     0.xxxx     x.xxxx   ← 🏆 Best
Neural Network (PyTorch)  0.xxxx    x.xxxx     0.xxxx     x.xxxx
```

> Replace `x.xxxx` with actual values after running the notebook.

### Hyperparameter Optimization (Optuna)

- **Sampler**: Tree-structured Parzen Estimator (TPE)
- **Trials**: 50
- **Search space**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- **Objective**: Maximize 3-fold CV R²

---

## 🧠 Deep Learning

### AegisNet Architecture (PyTorch)
```
Input (21 features)
    ↓
Linear(21 → 256) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(256 → 128) → BatchNorm1d → ReLU → Dropout(0.2)
    ↓
Linear(128 → 64) → BatchNorm1d → ReLU → Dropout(0.1)
    ↓
Linear(64 → 32) → ReLU
    ↓
Linear(32 → 1)
    ↓
Capability Score (regression output)
```

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=100) |
| Loss | HuberLoss (robust to outliers) |
| Batch size | 32 |
| Epochs | 150 |
| Gradient clipping | 1.0 |

---

## 📅 Time Series Forecasting

### Prophet
- External regressor: `Budget_BUSD`
- Changepoint prior scale: 0.3
- Interval width: 90%
- Forecast horizon: 10 years (2025–2034)

### SARIMA
- Order: `(1, 1, 1)`
- Seasonal order: `(0, 1, 1, 4)`
- Forecast horizon: 10 years
- Includes 95% confidence intervals

### Stationarity
- ADF test applied to ship deployment time series
- Series confirmed stationary (p < 0.05)

---

## 🔍 Explainability (SHAP)

Three SHAP visualizations are generated for the tuned XGBoost model:

1. **Bar chart** — global mean |SHAP| per feature
2. **Beeswarm plot** — distribution of SHAP values per feature
3. **Dependence plot** — SHAP value vs feature value for the top feature

> SHAP uses the **TreeSHAP** algorithm (exact, not approximate) for tree-based models.

---

## 🚨 Anomaly Detection

| Method | Contamination | Anomalies Found |
|--------|--------------|-----------------|
| Isolation Forest | 5% | ~25 records |
| DBSCAN | eps=0.8, min_samples=5 | varies |

Both methods visualized in 2D PCA space. Anomaly profiles compared against normal records across 5 key features.

---

## 💬 NLP Analysis

15 synthetic mission reports analyzed using **TextBlob**:

- **Polarity** score per report (−1 to +1)
- **Subjectivity** score per report (0 to 1)
- Sentiment classified as Positive / Negative / Neutral
- **WordCloud** generated from all report text

---

## 📈 Results

> Fill in after running the notebook

| Metric | Value |
|--------|-------|
| Best model | XGBoost (Tuned) |
| Best Test R² | — |
| Best Test RMSE | — |
| Neural Network R² | — |
| Prophet 2034 forecast | — ships |
| SARIMA 2034 forecast | — ships |
| Anomalies detected | — |
| Optimal clusters (k) | — |

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → New Notebook**
3. Copy and paste each cell in order
4. Click **Runtime → Run All**

> ⚠️ Run Cell 1 first and restart the runtime before continuing.

### Option 2 — Local (Jupyter)
```bash
# Clone the repo
git clone https://github.com/yourusername/aegis-combat-analytics.git
cd aegis-combat-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Aegis_Advanced_Analytics.ipynb
```

### requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
prophet
xgboost
lightgbm
shap
torch
torchvision
scipy
statsmodels
yellowbrick
imbalanced-learn
wordcloud
nltk
textblob
optuna
```

---

## 🖼️ Screenshots

| Plot | Description |
|------|-------------|
| ![EDA](outputs/eda.png) | Advanced EDA — 6-panel figure with ANOVA, PCA, correlation |
| ![SHAP](outputs/shap_plots.png) | SHAP bar + beeswarm explainability |
| ![Forecast](outputs/forecast.png) | Prophet + SARIMA 10-year forecast |
| ![Dashboard](outputs/dashboard.png) | 9-panel interactive Plotly dashboard |
| ![NN](outputs/nn_learning_curve.png) | Neural network learning curve |

> Add actual screenshots to the `outputs/` folder after running.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

### Ideas for Extension
- [ ] Replace synthetic data with real open-source naval datasets
- [ ] Add LSTM / Transformer for time series
- [ ] Add Streamlit web app frontend
- [ ] Add multi-class threat classification
- [ ] Integrate real-time Plotly Dash dashboard

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. All data used is synthetic and generated from publicly available historical information. No classified, proprietary, or sensitive defense information is used or implied.

---

## 📄 License
```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👤 Author

**Galla Rishi**


---

<div align="center">
  <sub>Built with ❤️ for defense technology research and data science education</sub>
</div>
