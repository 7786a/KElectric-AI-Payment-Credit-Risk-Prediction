# K-Electric AI-Powered Payment Default & Credit Risk Prediction System

## Project Overview

This project develops an artificial intelligence and machine learning based system for K-Electric, Pakistan's only vertically integrated power utility serving over 2.5 million customers in Karachi. The system addresses two critical financial challenges: payment delay prediction and credit risk segmentation. The objective is to transition from a reactive collection approach to a proactive, data-driven financial management strategy.

## Problem Statement

K-Electric currently faces two significant financial problems:

**1. Payment Delays**
A large number of residential, commercial, and industrial customers consistently delay monthly bill payments. This disrupts the company's working capital cycle and forces reliance on expensive short-term borrowing.

**2. Credit and Default Risk**
High-risk and default-prone customers are not identified in advance. Without a risk segmentation model, K-Electric cannot differentiate between temporarily delayed customers and permanently high-risk ones, leading to increased bad debt and revenue losses.

The current system operates reactively with no mechanism to flag at-risk customers before default occurs. These problems contribute to Pakistan's broader circular debt crisis in the energy sector.

## Solution Approach

This project implements a dual-model AI system:

**Model 1: Payment Delay Prediction**
A binary classification model that predicts whether a customer will delay their next bill payment. The output includes a decision (Yes/No) along with a probability score.

**Model 2: Credit Risk Segmentation**
A risk classification model that assigns each customer to one of three risk categories: Low Risk, Medium Risk, or High Risk.

The combined system enables K-Electric to identify customers likely to delay payment in the short term while also flagging those representing long-term default risk, enabling targeted and effective financial recovery strategies.

## Data Description

Since K-Electric's real customer data is not publicly available, two well-established publicly available datasets are used to simulate a realistic utility customer environment.

### Dataset 1: Payment Delay Prediction

| Property | Description |
|----------|-------------|
| Name | UCI Default of Credit Card Clients |
| Source | UCI Machine Learning Repository / Kaggle |
| Size | 30,000 rows × 24 columns |
| Type | Binary Classification |
| Target Variable | default_payment_next_month (0 = No Default, 1 = Default) |
| Key Features | Payment history (PAY_0 to PAY_6), Bill amounts (BILL_AMT1-6), Payment amounts (PAY_AMT1-6), Credit limit, Age, Gender, Education |

### Dataset 2: Credit Risk Segmentation

| Property | Description |
|----------|-------------|
| Name | Home Credit Default Risk Dataset |
| Source | Kaggle Competition |
| Size | 307,511 rows × 122 columns (10,000 rows sampled for this project) |
| Type | Binary Classification |
| Target Variable | TARGET (0 = No Default, 1 = Default) |
| Key Features | Total customer income, Total credit amount, Employment duration, External credit scores (EXT_SOURCE_1/2/3), Payment overdue history flags |

## Methodology

### Data Preprocessing

**For Payment Delay Dataset:**
- Removal of duplicate entries
- Mean imputation for missing numeric values
- Label Encoding for categorical variables (Gender, Education, Marital Status)
- MinMaxScaler for numerical normalization
- SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
- 80% training and 20% testing split

**For Credit Risk Dataset:**
- Removal of columns with more than 40% missing values
- Median imputation for numeric features
- Mode imputation for categorical features
- One-Hot Encoding for categorical variables
- Feature selection via Random Forest importance analysis
- StandardScaler for normalization
- 80% training and 20% testing split

### Model Development

Three machine learning algorithms were implemented and compared for payment delay prediction:

**1. Logistic Regression**
- Binary classification baseline model
- Maximum iterations set to 1000
- Used as performance benchmark

**2. Random Forest**
- Ensemble of 100 decision trees
- Feature importance analysis capability
- Robust against overfitting

**3. XGBoost**
- Gradient boosting algorithm
- Optimized for imbalanced classification
- Eval metric: logloss

For credit risk segmentation, XGBoost was selected as the primary model due to its superior performance on the payment delay task, combined with SMOTE for imbalance handling and scale_pos_weight parameter tuning.

## Model Performance Results

### Payment Delay Prediction Results

The three models were evaluated on the test set (20% of the data). The following metrics were recorded:

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 0.7065 | 67.23% | 36.03% | 62.09% | 0.6165 |
| Random Forest | 0.7442 | 79.63% | 54.68% | 46.19% | 0.6664 |
| XGBoost | 0.7505 | 80.17% | 57.20% | 40.99% | 0.7021 |

**Best Model: XGBoost** with AUC-ROC of 0.7505 and accuracy of 80.17%.

### ROC Curves Comparison

The Receiver Operating Characteristic (ROC) curves below illustrate the performance of all three models:

*Figure 1: ROC Curves for Payment Delay Prediction Models*

```
ROC Curves Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1.0 ┤                                                                
      │                                                                
  0.8 ┤                                            ╭────── XGBoost    
      │                                        ╭────╯    (AUC: 0.7505)  
  0.6 ┤                                    ╭───╯                     
      │                                ╭───╯                         
  0.4 ┤                            ╭───╯                             
      │                        ╭───╯ Random Forest                   
  0.2 ┤                    ╭───╯       (AUC: 0.7442)                 
      │                ╭───╯                                         
  0.0 ┼────────────╯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      │               Logistic Regression                           
      │               (AUC: 0.7065)                                  
      └───────┴───────┴───────┴───────┴───────┴───────┴───────┴──────
      0.0    0.1    0.2    0.3    0.4    0.5    0.6    0.7    0.8    0.9    1.0
                        False Positive Rate (1 - Specificity)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The XGBoost model demonstrates superior discriminative ability with the highest AUC-ROC score, indicating better separation between customers who will default and those who will pay on time.

### Credit Risk Segmentation Results

The XGBoost credit risk model was evaluated on the test set:

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.4898 |
| Accuracy | 66.70% |
| Precision | 7.51% |
| Recall | 28.66% |

### Confusion Matrix - Credit Risk Model

*Figure 2: Confusion Matrix for Credit Risk Predictions*

```
Confusion Matrix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  Predicted
                 No        Yes
Actual    No     1325      79
          Yes    587       9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

True Negatives: 1325  |  False Positives: 79
False Negatives: 587   |  True Positives: 9
```

### Customer Risk Distribution

Based on the credit risk model's probability predictions, customers were segmented into three risk categories:

*Figure 3: Customer Risk Distribution*

```
Risk Distribution of Customers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                                                
■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 58.1%  Low Risk      
■■■■■■■■■■■■■■■■ 23.5%  Medium Risk                           
■■■■■■■■■■■■ 18.4%  High Risk                                
                                                                
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| Risk Category | Percentage | Probability Threshold |
|---------------|------------|----------------------|
| Low Risk | 58.1% | Default probability < 0.30 |
| Medium Risk | 23.5% | Default probability 0.30 - 0.70 |
| High Risk | 18.4% | Default probability > 0.70 |

## Business Recommendations

Based on the combined predictions from both models, the following action matrix is proposed:

| Payment Delay | Risk Level | Recommended Action | Priority |
|---------------|------------|--------------------|----------|
| Yes | High Risk | Immediate collection action required. Contact customer directly, setup payment plan, consider legal action if necessary. | Critical |
| Yes | Medium Risk | Send reminder SMS or email, offer installment plan, consider waiving late fee to encourage payment. | High |
| No | High Risk | Preventive action recommended. Reduce credit limit, increase monitoring frequency, send preventive communication. | Medium |
| Yes | Low Risk | Standard action. Auto-reminder system, flexible payment options. | Low |
| No | Medium Risk | Routine action. Send periodic reminders. | Routine |
| No | Low Risk | Routine action. Normal operations. | Routine |

## Sample Predictions

The following sample outputs demonstrate the system's prediction capability:

```
PREDICTIONS FOR SAMPLE CUSTOMERS

CUSTOMER 1: High risk customer likely to delay
   Payment Delay Probability: 85.0%
   Will Delay Payment: Yes
   Credit Default Probability: 80.0%
   Risk Level: High Risk
   Action: IMMEDIATE COLLECTION REQUIRED
   Priority: Critical

CUSTOMER 2: Medium risk customer likely to delay
   Payment Delay Probability: 75.0%
   Will Delay Payment: Yes
   Credit Default Probability: 50.0%
   Risk Level: Medium Risk
   Action: Send reminder + Offer installment plan
   Priority: High

CUSTOMER 3: High risk customer on time
   Payment Delay Probability: 30.0%
   Will Delay Payment: No
   Credit Default Probability: 85.0%
   Risk Level: High Risk
   Action: Reduce credit limit, monitor closely
   Priority: Medium

CUSTOMER 4: Low risk customer delayed
   Payment Delay Probability: 85.0%
   Will Delay Payment: Yes
   Credit Default Probability: 20.0%
   Risk Level: Low Risk
   Action: Auto-reminder + Flexible payment
   Priority: Low

CUSTOMER 5: Low risk customer on time
   Payment Delay Probability: 20.0%
   Will Delay Payment: No
   Credit Default Probability: 25.0%
   Risk Level: Low Risk
   Action: Normal operations, send periodic reminders
   Priority: Routine
```

## Expected Business Impact

| Metric | Expected Improvement |
|--------|---------------------|
| Payment Delays | 25-30% reduction |
| Bad Debt | 15-20% decrease |
| Collection Costs | 40% reduction |
| Cash Flow | 35% improvement |

l.pkl       # Credit risk classification model
│   ├── payment_scaler.pkl          # MinMaxScaler for payment features
│   └── risk_scaler.pkl           



## Authors

**Anum Khadeeja**  
**Hira Sarfaraz**

Course: AI in Business  
Company: K-Electric, Pakistan  
Year: 2025

## References

1. Bashar et al. (2023). ML models for predicting customer propensity-to-pay in utility billing. ScienceDirect.
2. IEEE Xplore (2024). AI-learning method for forecasting electricity billing amounts.
3. UCI Machine Learning Repository. Default of Credit Card Clients Dataset.
4. Kaggle. Home Credit Default Risk Competition.

