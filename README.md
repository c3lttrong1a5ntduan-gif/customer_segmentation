# 🎯 E-Commerce Customer Segmentation via Behavioral Features, K-Means++, and SHAP Interpretability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-lightgrey.svg)](https://archive.ics.uci.edu/dataset/352/online+retail)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

An end-to-end Machine Learning and Explainable AI (XAI) framework designed to segment e-commerce customers based on 14 advanced behavioral features. This project extends beyond traditional RFM modeling by incorporating return behaviors, basket heterogeneity, and seasonal concentration. It utilizes a two-stage clustering framework combined with a SHAP-based surrogate model to derive transparent, ROI-driven business strategies.

---

## 📌 Executive Summary & Key Results

The pipeline employs a **Two-Stage Clustering Strategy**:
1. **Stage 1 (Rule-Based Filtering):** Identifies and separates 179 wholesale-like accounts (characterized by `SKU_HHI > 0.5`, representing ~4.6% of the customer base).
2. **Stage 2 (K-Means++ with k=4):** Segments the remaining 3,741 retail customers using 14 engineered behavioral features scaled via `QuantileTransform` + `StandardScaler`.

### Customer Cohort Overview (The Pareto Paradox)

| Cluster | Cohort Name | Population (%) | Revenue Share (%) | Median LTV (£) | Key Behavioral Signals & Strategic Action |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **C1** | **High-Return Actives** | 1,107 (29.6%) | **~71.0%** | **£1,972** | High `ReturnRate` (~25%), high order frequency. *Strategy: VIP retention & return friction reduction.* |
| **C3** | **Occasional Loyalists** | 1,047 (28.0%) | ~19.0% | £859 | Zero returns, high account age, steady purchasing. *Strategy: Loyalty rewards & basket upselling.* |
| **C0** | **Seasonal Intensives** | 330 (8.8%) | ~4.0% | £605 | High `MonthlyOrderRate` & `QuarterConcentration` (Q4 peak spenders). *Strategy: Pre-season triggers.* |
| **C2** | **One-Time Explorers** | 1,257 (33.6%) | ~6.0% | £265 | Low `BasketSizeCV`, high `Recency`, single transaction. *Strategy: Automated low-cost reactivation.* |

> **Key Insight:** C1 represents only ~30% of the customer base but generates **~71% of total revenue**. Mitigating returns in C1 provides the single highest financial leverage for the business.

---

## 📊 Model Explainability (SHAP Value Decomposition)

To decode the "black box" of unsupervised cluster boundaries, a **Surrogate Random Forest Classifier** (200 trees, **99.0% 5-fold OOF Accuracy**) was trained on cluster labels to extract SHAP TreeExplainer attributions.

* **Global Drivers:** `ReturnRate` accounts for **~27%** of global feature importance, followed by `QuarterConcentration` (~21%) and `BasketSizeCV` (~18%).
* **RFM Limitation Exposure:** Traditional RFM misclassifies C1 as standard "VIPs" (ignoring heavy cancellation costs) and mislabels C2 as "At-Risk" (triggering unnecessary, expensive discount offers).

*(Place your SHAP Summary Plot image here)*  
`![SHAP Beeswarm Plot](figures/04_shap/shap_beeswarm_retail_k4.png)`

---

## ⚙️ Repository Structure

```text
customer-segmentation-shap/
├── notebooks/                     # Sequential analysis notebooks
│   ├── 01_cleaning_and_eda.ipynb  # Data cleaning & transaction-level EDA
│   ├── 02_feature_engineering.ipynb # Engineering 14 behavioral features
│   ├── 03_modeling.ipynb         # Two-stage K-Means++ clustering & evaluation
│   ├── 04_shap_analysis.ipynb    # SHAP surrogate modeling & feature contributions
│   └── 05_cluster_profiling.ipynb # Business profiling & cohort strategies
├── src/                           # Production-style Python modules
│   ├── clustering_library/        # Core DataCleaner, FeatureEngineer, & ClusterAnalyzer
│   ├── notebook_io.py             # File path and I/O utilities
│   └── visual_style.py            # Consistent visualization design system
├── data/                          # Managed via .gitignore
│   ├── raw/                       # Raw online_retail.csv
│   └── processed/                 # Processed CSV feature tables and outputs
├── figures/                       # Saved high-resolution SHAP & profiling plots
├── pyproject.toml                 # Project metadata & dependencies
├── requirements.txt               # Locked dependencies list
├── .gitignore
