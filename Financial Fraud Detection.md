#### Financial Fraud Detection Using Machine Learning



###### Overview:

This project develops a machine learning solution to detect fraudulent financial transactions using real-world transaction data. The objective is to build a robust model that accurately identifies fraud while minimizing false positives to ensure a balance between financial security and customer experience.



###### Business Problem:

Financial fraud is rare but highly impactful. In transaction datasets, fraudulent cases represent a very small proportion, leading to \*\*class imbalance\*\*, which makes detection challenging.



The key objectives of this project:

\- Improve fraud detection (high recall)

\- Reduce false positives (customer inconvenience)

\- Build a scalable and interpretable solution



###### Dataset:

\- Credit card transaction dataset (anonymized)

\- Features:

  - V1–V28: PCA-transformed variables

  - Amount: Transaction value

  - Time: Transaction timestamp

  - Class:

    - 0 → Normal transaction

    - 1 → Fraudulent transaction



Dataset not included due to size

Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud



###### Project Workflow:



1\. Data Understanding

\- Analyzed dataset structure and feature distribution

\- Identified severe class imbalance



2\. Exploratory Data Analysis (EDA)

\- Class distribution analysis

\- Transaction amount distribution

\- Fraud behavior exploration



3\. Data Preprocessing

\- Feature scaling (Amount, Time)

\- Train-test split

\- Handling imbalance using SMOTE



4\. Model Development

\- Logistic Regression (Baseline)

\- Logistic Regression with SMOTE

\- Random Forest Classifier



5\. Model Evaluation

Evaluated using:

\- Confusion Matrix

\- Precision

\- Recall

\- F1-score



###### Model Performance Summary:



|Model|Precision|Recall|Interpretation|
|-|-|-|-|
|Logistic Regression|High|Moderate|Misses fraud cases|
|SMOTE + Logical Regression|Low|High|Detects fraud but many false alarms|
|Random Forest|Balanced|Balanced|Best Overall performance|

###### 



###### Final Model — Random Forest:



The Random Forest model delivered the best performance:



\- Precision: \*\*0.83\*\*

\- Recall: \*\*0.83\*\*

\- Low false positives

\- Strong balance between fraud detection and customer experience



###### Key Insights:



\- Class imbalance significantly impacts model performance

\- SMOTE improves fraud detection but increases false positives

\- Random Forest captures complex patterns better than linear models

\- Fraud detection is driven by a small set of key features



###### Feature Importance:



Top contributing features identified by the model:

\- V14

\- V10

\- V4

\- V12



These variables play a critical role in detecting fraudulent transactions.



###### Tools \& Technologies:



\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- imbalanced-learn

\- Jupyter Notebook



###### How to Run:



1\. Clone the repository

2\. Install dependencies:

   ```bash

   pip install -r requirements.txt

