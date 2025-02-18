# Opioid Receptor Ligand Binding

## Overview

This repository contains computational tools and machine learning models for predicting ligand binding efficiency to the Mu opioid receptor. The project applies molecular docking and machine learning techniques to identify novel opioid receptor ligands with potential therapeutic applications.

The primary objectives of this study are:
1. **Ligand Binding Prediction**: Utilize machine learning models to predict the efficacy of various compounds in binding to the Mu opioid receptor.
2. **Molecular Docking**: Perform molecular docking simulations using *SMINA* to assess ligand-receptor interactions.
3. **Similarity-Based Compound Discovery**: Expand the chemical search space by querying structurally similar compounds and evaluating their docking affinities.
4. **Model Performance Optimization**: Use various regression and classification models to enhance the prediction of ligand efficacy.

## Project Workflow

The research is divided into two primary phases:

### 1. Machine Learning-Based Ligand Prediction
- **Dataset Preparation**:
  - The dataset comprises 10,000 commercial compounds sourced from ChEMBL.
  - Each compound is represented using SMILES strings and transformed into **MACCS keys fingerprints**.
  - *Lipinski's Rule of Five* is applied to filter out non-drug-like molecules.
  - Compounds are further screened using PAINS filters to remove non-specific binders.
  - The remaining compounds are annotated with **pIC50 values** (negative log of IC50), a measure of binding potency.

- **Machine Learning Models**:
  - The following models are trained on chemical fingerprints to predict pIC50:
    - **Classification Models**: Random Forest, Support Vector Machines (SVM), Artificial Neural Networks (ANN).
    - **Regression Models**: Linear Regression, Ridge Regression, Lasso Regression, Decision Trees, Random Forest Regressors, Gradient Boosting, K-Nearest Neighbors (KNN), and XGBoost.
  - Model performance is evaluated using:
    - **ROC Curves** (for classification)
    - **R², MAE, RMSE** (for regression)

- **Feature Engineering**:
  - Various chemical fingerprints (MACCS, Morgan2, Morgan3) are tested.
  - The models initially predict binary activity (active/inactive) but later transition to **continuous pIC50 prediction**, increasing accuracy by over 30%.

### 2. Molecular Docking & Hit Expansion
- **Protein Preparation**:
  - The **Mu opioid receptor structure (PDB ID: 5C1M)** is obtained from the Protein Data Bank.
  - The receptor and ligand structures are converted into **PDBQT** format for docking.

- **Docking Process**:
  - The docking software *SMINA* is used to determine the **binding affinity** between ligands and the receptor.
  - The ligand with the highest docking score is used to conduct **iterative similarity searches** in PubChem.

- **Hit Expansion via Similarity Search**:
  - A **75% similarity threshold** is used to identify structurally related compounds.
  - The top hits are re-docked, and the most promising compounds are iteratively expanded into larger search spaces.

## Dependencies

To run the scripts, install the following dependencies:

```bash
pip install pandas rdkit scikit-learn tqdm matplotlib openbabel MDAnalysis xgboost
```

## How to Use

### 1. Train the Machine Learning Model
```bash
python ML\ Model\ Training.py
```
This script preprocesses the dataset, filters non-drug-like compounds, and trains classification-based ML models.

To use the advanced continuous ML models:
```bash
python Continuous+Expanded_ML_Models.py
```
This script trains regression models to predict pIC50 values.

### 2. Perform Molecular Docking

### 3. Expand the Chemical Search Space
- Modify `Docking_loop.py` to include newly identified high-affinity ligands in the **PubChem similarity search**.
- Run additional docking simulations iteratively.

## Results & Future Work
- The best-performing ML model achieved **high predictive accuracy (> 80% ROC AUC for classification, R² > 0.7 for regression)**.
- Iterative similarity-based docking revealed **novel high-affinity compounds**.
- Future improvements:
  - **Incorporate deep learning models** (e.g., Graph Neural Networks) for enhanced feature extraction.
  - **Expand hit exploration** using more iterations and alternative docking software.
  - **Integrate experimental validation** to confirm computational predictions.
