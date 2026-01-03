![DeepSurv Breast Prognosis Banner](deep_surv_banner.png)

<div align="center">

# DeepSurv-Breast-Prognosis
**AI-Powered Survival Analysis for Breast Cancer Precision Medicine**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Scikit-Survival](https://img.shields.io/badge/Scikit--Survival-ML-green?style=for-the-badge)](https://scikit-survival.readthedocs.io/en/stable/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 🏆 Team ByteRunners
**Official Submission for BioFusion Hackathon 2026**

| Member | Role |
| :--- | :--- |
| **Pasan** | AI Research & Model Architecture |
| **Nimsara** | Data Engineering & Visualization |
| **Theniya** | Biomedical Domain Analysis |

---

## 📖 Overview
**DeepSurv-Breast-Prognosis** is a state-of-the-art survival analysis pipeline designed to predict patient risk and survival probability. By leveraging both **Random Survival Forests (RSF)** and **DeepSurv (Deep Neural Networks for Cox Proportional Hazards)**, our solution moves beyond simple binary classification to provide time-to-event predictions essential for clinical decision-making.

This project utilizes the **METABRIC** dataset to identify high-risk genomic and clinical features, offering interpretable insights for personalized treatment strategies.

### 🚀 Key Features
- **Dual-Model Approach**: Benchmarks Random Survival Forests against Deep Neural Networks.
- **Time-to-Event Prediction**: Estimates survival probability over continuous time (e.g., 5-year survival).
- **Interpretability**: Permutation importance analysis to identify key risk drivers.
- **Robust Preprocessing**: Handles missing clinical data and right-censored survival targets.

---

## 📊 Performance
Our models have been rigorously evaluated using the **Concordance Index (C-Index)**, the gold standard for survival analysis.

| Model | C-Index (Test) | Improvement vs Random (0.5) |
| :--- | :---: | :---: |
| **Random Survival Forest** | **0.6641** | **+32.8%** |
| **DeepSurv (MLP)** | **0.6612** | **+32.2%** |

> *A C-Index of 0.66 indicates a strong ability to correctly rank patient survival times, significantly outperforming random guessing.*

---

## 🧬 Dataset
We utilize the **Molecular Taxonomy of Breast Cancer International Consortium (METABRIC)** dataset.
- **Source**: [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)
- **Samples**: ~1,900 patients
- **Features**: Gene expression profiles + Clinical attributes (Age, Tumor Stage, treatments).
- **Target**: Overall Survival (Months) + Vital Status.

---

## 🛠 Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch
- Scikit-learn & Scikit-survival

### Setup
```bash
# Clone the repository
git clone https://github.com/ByteRunners/DeepSurv-Breast-Prognosis.git
cd DeepSurv-Breast-Prognosis

# Install dependencies
pip install torch scikit-survival pandas matplotlib seaborn
```

### Running the Analysis
1. **Model 1 (Random Survival Forest)**:
   ```bash
   # Run the notebook to train RSF and see permutation importance
   jupyter notebook model_1.ipynb
   ```
2. **Model 2 (DeepSurv)**:
   ```bash
   # Run the deep learning pipeline
   jupyter notebook model_2.ipynb
   ```

---

## 📈 Visualizations
The project generates critical medical insights:
- **Survival Curves**: Predicted survival probability over 200+ months for individual patients.
- **Feature Importance**: Identifying whether "Tumor Size" or "Age" contributes more to risk.

---

## 🤝 Acknowledgements
- **BioFusion Hackathon 2026** organizers.
- **cBioPortal** for open-access cancer genomics data.
- The open-source community behind `scikit-survival` and `pytorch`.
