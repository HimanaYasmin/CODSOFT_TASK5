# CODSOFT_TASK5

# 💳 Credit Card Fraud Detection using Machine Learning

Project Overview
Credit card fraud detection is a critical problem in the financial industry. This project aims to identify **fraudulent credit card transactions** using machine learning techniques. Since fraudulent transactions are extremely rare compared to genuine ones, the dataset is **highly imbalanced**, making this a challenging classification problem.



Objective
To build an efficient machine learning model that can correctly classify transactions as:
- **0 → Genuine Transaction**
- **1 → Fraudulent Transaction**

The primary focus is on **Recall**, as failing to detect fraud can result in significant financial loss.



 Dataset Description
- **Dataset Source:** Kaggle – Credit Card Fraud Detection
- **Total Transactions:** 284,807
- **Fraud Cases:** 492
- **Features:**
  - `Time` – Time elapsed between transactions
  - `V1` to `V28` – PCA transformed features
  - `Amount` – Transaction amount
  - `Class` – Target variable (0 = Genuine, 1 = Fraud)



 Problem Statement
- The dataset is **highly imbalanced** (Fraud ≈ 0.17%)
- Accuracy alone is not a reliable metric
- Special techniques are required to handle imbalance



 Tools & Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)



 Project Workflow
1. Data Loading and Understanding
2. Exploratory Data Analysis (EDA)
3. Feature Scaling (`Time` and `Amount`)
4. Train-Test Split (Stratified)
5. Handling Class Imbalance using **SMOTE**
6. Model Training
   - Logistic Regression
   - Random Forest Classifier
7. Model Evaluation
8. Model Comparison
9. Final Model Selection



 Machine Learning Models

 Logistic Regression
- Used as a baseline model
- Simple and interpretable

 Random Forest Classifier
- Handles complex patterns
- Performs better on imbalanced data
- Achieved higher **Recall** and **F1-Score**


 Evaluation Metrics
- Confusion Matrix
- Precision
- **Recall (Primary Metric)**
- F1-Score

*Recall is prioritized because detecting fraud is more important than overall accuracy.*



 Results Summary
| Model | Precision | Recall | F1-Score |
|------|----------|--------|----------|
| Logistic Regression | Good | Moderate | Good |
| Random Forest | Better | **High** | **Best** |

 **Random Forest** was selected as the final model.



 Conclusion
This project demonstrates the effective use of machine learning techniques to detect fraudulent credit card transactions. By addressing class imbalance with SMOTE and focusing on recall-based evaluation, the Random Forest model achieved strong fraud detection performance.



Author
Himana Yasmin K.
