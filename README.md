# Credit Card Fraud Detection

This project analyzes a dataset of credit card transactions to identify fraudulent behavior. Using various preprocessing techniques and machine learning models, the goal is to effectively distinguish between normal and fraudulent transactions.

---

## Dataset

The dataset used in this project is a credit card transactions dataset, which contains both fraudulent and non-fraudulent activities.

- **Source:** `creditcard.csv`  
- **Columns:**  
  - **Time:** The time elapsed since the first transaction.  
  - **Amount:** The transaction amount.  
  - **Class:** 0 for non-fraudulent transactions, 1 for fraudulent ones.

---

## Project Workflow

1. **Data Loading and Exploration:**  
   Importing the data using pandas and displaying the structure.

   ```python
   import pandas as pd
   df = pd.read_csv('/Users/yenhuynh/Downloads/creditcard.csv')
   df.head() 

2. **Class Distribution:**
   Checking the number of fraudulent vs non-fraudulent transactions.
   
   ```python
   df['Class'].value_counts()

3. **Data Visualization:**
   Plotting histograms to explore feature distributions.
   
   ```python
    df.hist(bins=30, figsize=(30,30));

4. **Descriptive Statistics:**
   Summarizing the numerical features using describe().

    ```python
    df.describe()

5. **Data Preprocessing:**
   Scaling and normalizing features.
   
   ```python
    from sklearn.preprocessing import RobustScaler

    new_df = df.copy()
    new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].values.reshape(-1,1))

    time = new_df['Time']
    new_df['Time'] = (time - time.min()) / (time.max() - time.min())

## Next Steps
 - Explore feature engineering techniques or anomaly detection methods.
 - Train machine learning models such as Logistic Regression or Random Forest.
 - Address class imbalance using techniques like SMOTE or under-sampling.

  
