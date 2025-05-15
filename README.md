# Flu Vaccine Uptake Prediction

This project predicts H1N1 and seasonal flu vaccine uptake using logistic regression and Light GBM machine learning models.

## Table of Contents

-   [1.  Business Understanding](#1-business-understanding)
-   [2.  Data Understanding](#2-data-understanding)
-   [3.  Data Preparation](#3-data-preparation)
-   [4.  Modeling](#4-modeling)
-   [5.  Evaluation](#5-evaluation)
-   [6.  Deployment](#6-deployment)
-   [Files](#files)
-   [Getting Started](#getting-started)

## 1.  Business Understanding

###   Objective

The goal is to develop predictive models that accurately estimate individuals’ vaccination probabilities to aid public health outreach and intervention efforts.

###   Motivation

Understanding the complex interplay of factors influencing vaccine uptake can help public health organizations design more effective campaigns and allocate resources efficiently.

## 2.  Data Understanding

###   Data Sources

The project uses data from the National 2009 H1N1 Flu Survey. The key datasets are:

-   `training_set_features.csv`: Contains independent variables (features) for training the models.
-   `training_set_labels.csv`: Contains the target variables (H1N1 and seasonal flu vaccine uptake) for training.
-   `test_set_features.csv`: Contains independent variables for which predictions are needed.
-   `submission_format.csv`: Specifies the required format for the submission file.

###   Data Description

-   Features: 36 columns representing various aspects of respondents, including:
    -   Beliefs/Concerns (e.g., `h1n1_concern`, `opinion_h1n1_vacc_effective`)
    -   Behaviors (e.g., `behavioral_antiviral_meds`, `behavioral_wash_hands`)
    -   Demographics (e.g., `age_group`, `education`, `income_poverty`)
    -   Health-Related (e.g., `chronic_med_condition`, `health_worker`)
-   Target Variables:
    -   `h1n1_vaccine`: Binary variable indicating H1N1 vaccine uptake (0 or 1).
    -   `seasonal_vaccine`: Binary variable indicating seasonal flu vaccine uptake (0 or 1).
-   Data Size:
    -   Training set: 26,707 samples, 36 features.
    -   Test set: 26,708 samples, 36 features.

- **Target Distributions**:
  - `h1n1_vaccine`: 21.25% vaccinated, 78.75% not vaccinated
  - `seasonal_vaccine`: 46.56% vaccinated, 53.44% not vaccinated

###   Exploratory Data Analysis (EDA)

The `Flu_Shot_Learning.ipynb` notebook performs EDA:

-   Loading and inspecting data:

    ```python
    import pandas as pd
    feat_train = pd.read_csv('Data/training_set_features.csv')
    print(feat_train.head())
    print(feat_train.info())
    ```

    * This provides a glimpse into the data structure and data types.
    * For example, `feat_train.info()` reveals that some columns have missing values, and `age_group` is of object type (categorical).

-   Missing value analysis:

    ```python
    print(feat_train.isnull().sum().sort_values(ascending=False).head())
    ```

    * The top 5 columns with the most missing values are:
        * `employment_occupation`: 13470 missing values
        * `employment_industry`: 13330 missing values
        * `health_insurance`: 12274 missing values
        * `income_poverty`: 4423 missing values
        * `doctor_recc_seasonal`: 2160 missing values

    * This highlights that employment-related information and health insurance coverage are frequently missing.

-   Target variable distribution:

    ```python
    print(lbl_train['h1n1_vaccine'].value_counts())
    print(lbl_train['seasonal_vaccine'].value_counts())
    ```

    * H1N1 vaccine uptake:
        * 0 (Not Vaccinated): 21033
        * 1 (Vaccinated): 5674
    * Seasonal vaccine uptake:
        * 0 (Not Vaccinated): 14272
        * 1 (Vaccinated): 12435

    * The target variables exhibit class imbalance, especially `h1n1_vaccine`, where the number of non-vaccinated individuals is much higher than the number of vaccinated individuals.

## 3.  Data Preparation

-   Feature engineering: A binary feature `has_underlying_condition` is created to indicate the presence of any chronic medical condition.
-   Missing value handling: Categorical features are filled with the string 'missing'.
-   Categorical encoding: Label Encoding is used to convert categorical features into numerical representations.
-   One-Hot Encoding: An example is shown for the 'education' feature to illustrate the technique.

## 4.  Modeling

-   Models: Logistic Regression, LightGBM.
-   Cross-validation: Stratified K-Fold cross-validation is used to handle class imbalance and provide more robust evaluation.
-   Logistic Regression:
    -   Separate models are trained for H1N1 and seasonal vaccine prediction.
    -   Class imbalance is addressed using the `class_weight='balanced'` parameter.
-   LightGBM:
    -   Separate models are trained for each target variable.
    -   AUC (Area Under the ROC Curve) is used as the primary evaluation metric.
    -   Model parameters are set as: `objective='binary'`, `metric='auc'`, `n_estimators=100`, `learning_rate=0.05`, `random_state=42`, `verbose=-1`, `n_jobs=-1`.

## 5.  Evaluation

-   Metrics: AUC, classification report, confusion matrix.
-   Logistic Regression:
    -   Performance is evaluated with and without class weights to assess the impact of this technique.
    -   Classification reports provide precision, recall, and F1-score for each class.
    -   Confusion matrices visualize true positives, true negatives, false positives, and false negatives.
-   LightGBM:
    -   AUC scores are reported for both H1N1 and seasonal vaccine predictions.

###   Key Evaluation Results (from notebook)

-   **LightGBM Performance (Out-of-Fold AUC):**
    -   H1N1: 0.8749
    -   Seasonal: 0.8605

    * LightGBM models demonstrate strong performance on both targets, indicating a good ability to distinguish between vaccinated and non-vaccinated individuals.

-   **Logistic Regression Performance:**
    -   Logistic Regression models show lower AUC scores compared to LightGBM.
    -   Class imbalance significantly affects Logistic Regression performance. Without class weighting, the models tend to perform better on the majority class (non-vaccinated) and poorly on the minority class (vaccinated).
    -   Using `class_weight='balanced'` improves recall for the minority class but often reduces precision for the majority class, highlighting the precision-recall trade-off.

## 6.  Deployment

-   Predictions are generated on the test set using the trained LightGBM models.
-   Predictions are rounded to two decimal places.
-   A submission file (`submission.csv`) is created with the respondent IDs and predicted probabilities for both vaccines.

## Files

-   `Flu_Shot_Learning.ipynb`: Jupyter Notebook containing the data analysis and modeling code.
-   `Data/`: Directory containing the datasets.
-   `submission.csv`: CSV file with the model predictions.

## Getting Started

1.  Ensure you have Python 3.x installed along with the required libraries (pandas, scikit-learn, lightgbm).
2.  Install the libraries using `pip install pandas scikit-learn lightgbm`.
3.  Place the datasets in the `Data/` directory.
4.  Run the `Flu_Shot_Learning.ipynb` notebook.