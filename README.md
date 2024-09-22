# Credit Card Fraud Detection Using RandomForestClassifier
This project demonstrates the use of machine learning to detect fraudulent credit card transactions. We use a RandomForestClassifier for classification and SMOTE (Synthetic Minority Oversampling Technique) to handle the class imbalance problem. The dataset is sourced from the Kaggle Credit Card Fraud Detection Dataset.

### Dataset
The dataset contains transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

### Columns:
- Time: Number of seconds elapsed between this transaction and the first transaction in the dataset.
* V1 to V28: The result of a PCA transformation on the transaction data to anonymize sensitive information.
- Amount: The transaction amount.
* Class: The label for fraud detection. 1 means fraud, 0 means legitimate transaction.
## Project Workflow
### 1. Data Preprocessing
- Normalize the Amount column using StandardScaler.
* Drop the Time column as it's not relevant for the model.
### 2. Handling Imbalanced Data
- Use SMOTE to oversample the minority class (fraud cases) to balance the dataset before training.
### 3. Model Training
- Train a RandomForestClassifier using the oversampled dataset.
* Split the dataset into training and testing sets.
### 4. Model Evaluation
* Confusion Matrix: Provides insights into the number of true positives, true negatives, false positives, and false negatives.
- Classification Report: Shows metrics like precision, recall, F1-score for each class.
- ROC Curve: Displays the model's performance across different threshold values.

## Output
![image](https://github.com/user-attachments/assets/c57fb03d-9455-4e65-9ee7-8559b1c35011)
![image](https://github.com/user-attachments/assets/43118288-a16c-4192-871f-157be148c600)
![image](https://github.com/user-attachments/assets/305128df-e830-4fb9-af78-b365badb3ce2)
