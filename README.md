# 🏠 Property Fraud Detection (NYC)

This project focuses on identifying potential tax fraud within New York City's 2010/11 property assessment dataset using unsupervised machine learning methods.

## Files in this Repository

- **`NY unsupervised fraud.ipynb`** – Jupyter notebook containing all the data cleaning, feature engineering, PCA, and anomaly detection logic.
- **`Final Report.pdf`** – Comprehensive summary of the project, including methodology, algorithms, results, and flagged fraud cases.
- **`Data Quality Report.pdf`** – Automated EDA output (generated via pandas-profiling) used for understanding the dataset's structure and spotting inconsistencies.

##  Project Summary

New York City loses millions to property tax fraud each year, often due to underreported property values, misclassifications, or incorrect usage declarations. This project aims to surface suspicious records from a dataset of over 1 million property entries.

##  What Was Done

### 1. Data Cleaning
- Removed ~26K public/government properties.
- Imputed missing values for ZIP codes, property values (`FULLVAL`, `AVLAND`, `AVTOT`), and physical dimensions (`LTFRONT`, `LTDEPTH`, `STORIES`, etc.) using grouped means/modes.

### 2. Feature Engineering
- Created 29 new variables representing:
  - Value-to-size ratios (`r1`–`r9`)
  - ZIP-normalized and TaxClass-normalized versions of those ratios
  - Combined metrics like `value_ratio` and `size_ratio` to flag internal inconsistencies

### 3. Dimensionality Reduction
- Standardized features and applied **Principal Component Analysis (PCA)** to reduce noise and correlations in the data

### 4. Anomaly Detection
- **Score 1**: Minkowski distance in PCA space
- **Score 2**: Autoencoder-based reconstruction error
- Final fraud score = average rank of both scores

### 5. Results
- Properties were sorted by fraud score
- Visual inspections (e.g. via Google Maps) confirmed major inconsistencies:
  - Suspiciously low land values
  - Missing or incorrect addresses
  - Impossible building dimensions
  - Properties with 0 lot depth or misaligned tax classes

##  Tools & Technologies Used

- **Python**, **Jupyter Notebook**
- `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`
- `pandas-profiling` for the data quality report
