# Retroviral-Wall-Challenge

**Official Code Repository for the Mandrake Bio "Retroviral Wall Challenge" on Kaggle.**

This repository contains four machine learning pipelines designed to predict Reverse Transcriptase (RT) prime editing activity and efficiency. The dataset is characterized by a small sample size ($N=57$) with high dimensionality (66 handcrafted features and 1280-dimensional ESM-2 embeddings). 

The primary objective of this codebase is to optimize for Out-Of-Distribution (OOD) generalization (Phase 2 wet-lab validation) without overfitting to the evolutionary lineage of the public dataset.

## Phase 1 Validation Results (Leave-One-Family-Out Cross-Validation)

All models were evaluated using Leave-One-Family-Out (LOFO) cross-validation on the `rt_family` variable. This simulates the Phase 2 objective of predicting on novel evolutionary lineages.

| Submission | Technical Description | PR-AUC | Weighted Spearman | Final CLS |
| :--- | :--- | :---: | :---: | :---: |
| **Submission 1** | Decoupled Pipeline + PCA-Reduced ESM-2 | 0.6487 | 0.4255 | **0.5139** |
| **Submission 2** | Handcrafted Features Only + Seed Ensembling | 0.6439 | 0.4041 | **0.4965** |
| **Submission 3** | Task-Specific Feature Routing + ESM-2 PCA | 0.6309 | 0.3975 | **0.4877** |
| **Submission 4** | L1 Linear Classifier + Heteroscedastic GP | 0.6386 | 0.3640 | **0.4637** |

---

## Submission Portfolio Details

To account for different types of distribution shifts in the Phase 2 dataset, we submit four models with distinct architectural constraints and inductive biases.

### Submission 1: Decoupled Predictor with PCA-Reduced ESM-2 (`submission_1.py`)
* **Features Used:** Handcrafted features and ESM-2 embeddings.
* **Architecture:** The 1280-dimensional ESM-2 vectors are compressed using `PCA(n_components=6)`. The pipeline uses a decoupled classification and regression approach.
* **Rationale:** Reduces the dimensionality of language model embeddings to extract general biophysical variance while limiting the capacity to memorize specific phylogenetic sequences.

### Submission 2: Handcrafted Features Only + Ensembling (`submission_2.py`)
* **Features Used:** Strictly handcrafted features. ESM-2 embeddings and `foldseek_TM_*` features are excluded.
* **Architecture:** Uses `add_indicator=True` in `SimpleImputer` to retain missing structural data as a boolean feature. The final predictions are averaged across three random seeds (42, 1337, 2026).
* **Rationale:** Eliminates evolutionary sequence leakage by disabling language model embeddings, relying exclusively on calculated 3D structural and physicochemical properties.

### Submission 3: Task-Specific Feature Routing (`submission_3.py`)
* **Features Used:** Handcrafted features and PCA-reduced ESM-2 embeddings.
* **Architecture:** Implements separate feature selection parameters for the dual objectives. The classification branch selects 10 features using `f_classif`, while the regression branch selects 15 features using `f_regression`.
* **Rationale:** Assumes that the structural markers correlating with binary enzyme activity (active vs. inactive) differ statistically from the continuous properties governing maximum editing efficiency.

### Submission 4: Linear Classification and Heteroscedastic GP (`submission_4.py`)
* **Features Used:** Handcrafted features and PCA-reduced ESM-2 embeddings.
* **Architecture:** Replaces tree-based ensemble classifiers with a sparse L1-regularized Logistic Regression. The regression branch utilizes a Gaussian Process where `noise_variance` is set to `0.1 / weights`.
* **Rationale:** The most heavily regularized model in the portfolio. The modified Gaussian Process mathematically models measurement variance, assuming lower measurement noise for high-efficiency enzymes and higher noise for inactive enzymes.

---

## Core Pipeline Components

All four submissions share the following baseline components designed to optimize the Kaggle Cross-Lineage Score (CLS):

1. **Metric-Aligned Regression ("Custom Estimators")**  
   Standard regressors optimize for Mean Squared Error. To align with the Weighted Spearman metric, custom wrapper classes were built for `Ridge`, `SVR`, and `GaussianProcessRegressor`. These transform the target variable via `np.log1p(y)` and apply `sample_weights = y_true + 0.01` during the `.fit()` step.
2. **Decoupled Expected Value Formulation**  
   Classification and regression pipelines are trained independently. The final predicted score is mathematically formulated as $P(\text{active})^2 \times \hat{y}_{\text{eff}}$. This optimizes PR-AUC by reducing false positive predictions while preserving the continuous ranking of true positives.
3. **Phase 1 and Phase 2 Inference Routing**  
   The prediction step is dynamically routed. If a sequence is present in the training data, the script outputs the exact LOFO Out-Of-Fold prediction. If the sequence is novel (Phase 2), it outputs the prediction from the pipeline refitted on 100% of the training data.

---

## Execution Instructions (Phase 2 Extraction)

All logic (imputation, scaling, feature selection, and inference) is encapsulated inside `scikit-learn` Pipeline and ColumnTransformer objects to ensure portability.

### Setup Requirements
```bash
# Install dependencies
pip install numpy pandas scikit-learn scipy
