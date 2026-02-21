# Drug-Protein Interaction ML

> Prediction of drug-protein interactions using Machine Learning on real BindingDB data.  
> Personal project combining knowledge from computer science, business administration, and biochemistry.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Description

This project implements a **complete virtual screening pipeline** to predict whether a small molecule (drug candidate) is likely to bind to a target protein. The model enables prioritization of compounds from libraries of thousands of molecules before performing costly experimental assays, significantly reducing both the time and cost of the drug discovery process.

**Dataset:** [BindingDB](https://www.bindingdb.org/) — public database with over 2.8M affinity measurements between molecules and proteins.

---

## Business Impact and Operational Efficiency

As a Business Administration student, this project was designed not only as a technical challenge but as an **asset optimization and risk reduction solution** in the pharmaceutical R&D value chain.

* **The "Funnel Filter":** In drug discovery, physical High-Throughput Screening is extremely costly. This pipeline acts as a critical filter, processing a virtual library of **100,000 compounds** to prioritize only the **500 candidates** with the highest probability of success.
* **Cost Savings (OPEX):** By identifying failures early (*Fail Fast*), it avoids investing **millions of euros** in *in vitro* laboratory assays destined to fail, optimizing the research budget.
* **Time-to-Market Reduction:** Accelerating the initial screening stage allows promising molecules to reach clinical phases sooner, maximizing the net present value (NPV) of intellectual property.

---

## Repository Structure

```
DrugProtein_ML/
├── Drug_Discovery_Training.ipynb   # Complete training pipeline
├── Drug_Discovery_Inference.ipynb  # Prediction on new library
├── app.py                          # Interactive virtual screening application
├── requeriments.txt
├── xgboost.pkl                     # Trained model ready to use
├── examples/
│   └── mi_libreria.csv             # Example CSV for testing inference
└── images/                         # Plots generated during training
```

---

## Interactive Application

The project includes an **interactive web application built with Streamlit** that allows direct use of the model on new molecules or complete chemical libraries.

### Online Demo

The application is deployed and can be tested directly here:

👉 https://molecular-predictive-discovery.streamlit.app/

The application allows you to:

- Analyze an individual molecule from its SMILES string
- Obtain binding probability and Binder / Non-Binder classification
- Visualize the molecular structure in 2D and 3D
- Calculate physicochemical properties and drug-likeness
- Evaluate the multivariate molecular profile
- Perform automated screening of libraries in CSV format
- Rank candidates and interactively analyze the best hits

This reproduces the real workflow of **computational virtual screening prior to experimental screening**.

### Run the Application
```bash
pip install -r requirements.txt  
streamlit run app.py
```

## Methodology

### Data Preprocessing
- Loading 500k rows from BindingDB with Ki, IC50, and Kd measurements
- Chemical cleaning with RDKit: desalting, SMILES canonicalization, sanitization
- Deduplication by canonical SMILES
- Binary labeling: **binder** if affinity < 1000 nM (1 µM), **non-binder** otherwise
- 1:1 balancing via undersampling → 109,028 final molecules

### Molecular Featurization
| Feature | Description | Dimensions |
|---|---|---|
| **Morgan FP (ECFP4)** | Circular fingerprint radius 2, gold standard in QSAR | 2048 bits |
| **Physicochemical Descriptors** | MolWt, LogP, HBA, HBD, TPSA, RotBonds, Aromatic, QED, HeavyAtoms, FracCSP3 | 10 |

### Rigorous Split — Scaffold Split
The dataset is split by **Murcko scaffold**: train and test sets contain entirely distinct chemical families, simulating the real-world scenario of predicting on novel compounds.

**Split audit (4 checks):**
- Scaffold overlap: **0%** (< 1% acceptable)
- Exact SMILES in common: **0**
- Tanimoto NN >= 0.85: **0%** (< 5% acceptable)
- Class drift train/test: **0.6%** (< 5% acceptable)

### Trained Models

#### 1. XGBoost + Morgan FP + Descriptors — Best Model
Industry standard in pharma for QSAR (Bender et al., 2022). Gradient boosting on sparse Morgan FP features with L1/L2 regularization and early stopping.

#### 2. ChemBERTa + XGBoost
768-dimensional embeddings from a transformer pretrained on 77M molecules (ZINC), used as input for XGBoost. The architecture avoids overfitting from an additional MLP.

#### 3. Ensemble (simple average)
Combination of both models with fixed equal weights. **Weights are not optimized on the test set** to avoid data leakage.

---

## Results

| Model | Train ROC-AUC | Test ROC-AUC | Gap | PR-AUC | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|
| **XGBoost** | 0.9333 | **0.8979** | 0.0354 | **0.8993** | **0.8199** | **0.6470** | **0.1291** |
| ChemBERTa+XGB | 0.8748 | 0.8357 | 0.0391 | 0.8283 | 0.7627 | 0.5238 | 0.1651 |
| Ensemble | 0.9211 | 0.8832 | 0.0379 | 0.8830 | 0.8071 | 0.6174 | 0.1404 |

> **XGBoost outperforms the ensemble** because ChemBERTa was pretrained on ZINC, which has a different chemical distribution than BindingDB, limiting the quality of its embeddings in this domain.
>
> All train/test gaps are below 0.05, confirming the absence of significant overfitting.

### ROC and Precision-Recall Curves

![ROC and PR Curves](images/roc_pr_curves.png)

### Overfitting Diagnosis and Calibration

![Overfitting and Calibration](images/overfit_calibration.png)

### Methodological Guarantees
- Scaffold split with 4 integrity checks
- Early stopping in all XGBoost models
- Ensemble with fixed weights (no optimization on test set)
- Probability calibration reported (Brier score)
- Train/test gap reported for each model
- Feature interpretability (bitInfo + substructure visualization)

---

## Interpretability

### Feature Importance — XGBoost

Morgan Fingerprints dominate the model with 98.8% of total importance. The only physicochemical descriptor in the top 20 is **NumHeavyAtoms**, which makes biological sense: larger molecules have more surface contact area with the protein.

![Feature Importance](images/feature_importance.png)

### Activating Substructures for the Most Important Bits

Using RDKit's `bitInfo`, each Morgan bit can be mapped to the chemical substructure that activates it, visualized directly on the molecule.

![Morgan Substructures](images/morgan_substructures.png)

---

## Molecular Visualizations

### Top 10 Binders with Highest Model Confidence

![Top 10 Binders](images/top10_binders.png)

### Typical Binder vs. Typical Non-Binder

![Binder vs No Binder](images/binder_vs_nobinder.png)

### Error Analysis — False Positives and False Negatives

FPs have very weak affinities (> 10,000 nM) and are structurally similar to known kinase inhibitors. FNs are ultra-potent (< 1 nM), structurally atypical, and underrepresented in the training data.

![Error Analysis](images/error_analysis.png)

![Error Molecules](images/error_molecules.png)

### Chemical Space — t-SNE (3,000 molecules from the test set)

2D projection of Morgan FPs showing how binders and errors are distributed across structural space.

![t-SNE Chemical Space](images/tsne_chemical_space.png)

---

## Requirements

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
# Core — required for both notebooks
rdkit>=2023.3.1
xgboost>=1.7.0
scikit-learn>=1.2.0
joblib>=1.2.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
streamlit>=1.32.0
py3Dmol>=2.0.0
stmol>=0.0.9


# Only required for ChemBERTa (training notebook)
transformers>=4.30.0
torch>=2.0.0
```

| Library | Recommended Version |
|---|---|
| Python | >= 3.9 |
| RDKit | >= 2023.x |
| XGBoost | >= 1.7 |
| scikit-learn | >= 1.2 |
| transformers | >= 4.x (only for ChemBERTa) |
| PyTorch | >= 2.0 (only for ChemBERTa) |

---

## How to Run

### Option A — Inference only (fast, no retraining)
The model is already trained in `models/xgboost.pkl`. You only need your compound library.

1. Prepare a CSV with a `smiles` column (and optionally a `label` column with 0/1 values)
2. Open `Drug_Discovery_Inference.ipynb`
3. Edit the configuration cell with the path to your CSV
4. Run all cells

**Output:** binding probability, BINDER/NON-BINDER classification, molecular visualizations, and a report.

### Option B — Full training (requires BindingDB)
1. Download `BindingDB_All.tsv` from [bindingdb.org](https://www.bindingdb.org/bind/chemsearch/marvin/Download.jsp) (~6 GB)
2. Place it in the project root alongside the notebook
3. Open `Drug_Discovery_Training.ipynb`
4. Run all cells (~45 min with GPU, ~90 min without GPU)

> **Recommended:** run on [Google Colab](https://colab.research.google.com/) with GPU enabled to reduce ChemBERTa training time.

---

## Input CSV Format (Inference)

```csv
smiles,name,label
CC(=O)Oc1ccccc1C(=O)O,Aspirin,0
Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1,Imatinib,1
```

| Column | Required | Description |
|---|---|---|
| `smiles` | Yes | SMILES string of the molecule |
| `name` | No | Compound name |
| `label` | No | Ground truth label (0/1) — enables performance metrics |

---

## Limitations

- The model predicts binding **in general**, not against a specific target protein. It is useful as a structural pre-filter before experimental screening.
- Trained with the publication bias of BindingDB (overrepresentation of actives and certain targets such as kinases).
- The Enrichment Factor is limited by the 1:1 balancing of the training dataset.

---

## Author

Personal project developed as a demonstration of the intersection between computer science, business administration, and computational biochemistry.

- **Stack:** Python, RDKit, XGBoost, HuggingFace Transformers, scikit-learn
- **Data:** BindingDB (public domain)
- **Environment:** Google Colab / Jupyter Notebook

---

## License

MIT License — free for academic and personal use.
