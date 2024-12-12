"""
Created on Sun Nov 17 2024

EDA ver.3

-------------------KEY IMPROVEMENT--------------------
* Missing Value
    - Filling '-' for categorical columns
---------------------------------------------------------

@author: Jiang
"""

# ========================================================================
#                            Step 1: Data Loading
# ========================================================================
print('---------------------------------------------------------')
print('Step 1: Data Loading')
# ------------------------------------------------------------------------
#                            import packages and data
# ------------------------------------------------------------------------
# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
from matplotlib import pyplot as plt

# Machine Learning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Warnings
import warnings
warnings.filterwarnings('ignore')


# Helper function
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

'''
Since for tree models, missing value are also treated as a unique value, we will use the unmodified dataset.
'''

print('---------------------------------------------------------')
print('Loading data')

train_df = pd.read_csv('train.csv')
print(train_df.shape)
train_df = reduce_mem_usage(train_df)
train_df.head()

test_df = pd.read_csv('test.csv')
print(test_df.shape)
test_df = reduce_mem_usage(test_df)
test_df.head()

# Calculate the length of the original training data
len_train_df = len(train_df)

# Duplicates check in train data
train_df.duplicated().sum()

# # Class imbalance check
# plt.pie(test_df.isFraud.value_counts(), labels=['Not Fraud', 'Fraud'], autopct='%0.1f%%')
# plt.axis('equal')
# plt.show()
print('Data Distribution: 7.8% Fraud, 92.2% Not Fraud')

'''
As one can expect, this is a class imbalance problem. 
We will apply SMOTE (Synthetic Minority Over-sampling Technique) to deal with class imbalance in later steps. 
------
Let us understand the distribution of the timestamp column.
'''
# Timestamp of train and test data
plt.figure(figsize=(8, 4))
plt.hist(train_df['TransactionDT'], label='Train')
plt.hist(test_df['TransactionDT'], label='Test')
plt.ylabel('Count')
plt.title('Transaction Timestamp')
plt.legend()
plt.tight_layout()
plt.show()
''' We can notice that the timestamp of the test data is ahead of the timestamp of the train data. Therefore, 
while training machine learning model, we need to perform time-based splitting to create training and validation sets. 

Let us deal with the missing values first.
There are considerable number of columns with high missing values. 
We'll use only those columns that has at least 80% data which leaves 20% to the missing values that can be filled.
'''
print("Missing values check")
combined_df = pd.concat([train_df.drop(columns=['isFraud', 'TransactionID']), test_df.drop(columns='TransactionID')])
print(combined_df.shape)

# Dependent variable
y = train_df['isFraud']
print(y.shape)

# Dropping columns with more than 20% missing values
mv = combined_df.isnull().sum()/len(combined_df)
combined_mv_df = combined_df.drop(columns=mv[mv>0.2].index)
del combined_df, train_df, test_df
print(combined_mv_df.shape)

'''
We are left with 245 columns out of 433 after removing features with more than 20% missing values. 
We also have removed 'TransactionID' column as it does not hold any importance in the prediction. 

Let us now fill all the missing values. 
For numerical columns, 
we will use median value; 
for categorical column, 
we will use the most frequent category to fill the missing values.'''

# Filtering numerical data
num_mv_df = combined_mv_df.select_dtypes(include=np.number)
print(num_mv_df.shape)

# Filtering categorical data
cat_mv_df = combined_mv_df.select_dtypes(exclude=np.number)
print(cat_mv_df.shape)
del combined_mv_df

# Filling missing values by median for numerical columns
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
num_df = pd.DataFrame(imp_median.fit_transform(num_mv_df), columns=num_mv_df.columns)
del num_mv_df
print(num_df.shape)

# Filling missing values by '-' for categorical columns
imp_max = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value='-')
cat_df = pd.DataFrame(imp_max.fit_transform(cat_mv_df), columns=cat_mv_df.columns)
del cat_mv_df
print(cat_df.shape)

# Concatinating numerical and categorical data
combined_df_cleaned = pd.concat([num_df, cat_df], axis=1)
del num_df, cat_df

# Verifying missing values
print(f'Total missing values: {combined_df_cleaned.isnull().sum().sum()}')
print(combined_df_cleaned.shape)
combined_df_cleaned.head()
# ========================================================================
#                            Step 2: Data Preprocessing
# ========================================================================
print('---------------------------------------------------------')
print('Step 2: Data Preprocessing')

# One-hot encoding
combined_df_encoded = pd.get_dummies(combined_df_cleaned, drop_first=True)
print(combined_df_encoded.shape)
del combined_df_cleaned
combined_df_encoded.head()

# Separating train and test data
X = combined_df_encoded.iloc[:len_train_df, :]  # Include all columns
print(X.shape)
test = combined_df_encoded.iloc[len_train_df:, :]  # Include all columns
print(test.shape)

# Time-based train validation splitting with 20% data in validation set
train = pd.concat([X, y], axis=1)
train.sort_values('TransactionDT', inplace=True) # Sorts the data by the column TransactionDT, indicating transaction time.
X = train.drop(['isFraud'], axis=1)
y = train['isFraud']
splitting_index = int(0.8*len(X))
X_train = X.iloc[:splitting_index].values # Training Set and validation set
X_val = X.iloc[splitting_index:].values
y_train = y.iloc[:splitting_index].values
y_val = y.iloc[splitting_index:].values
test = test.values # Convert Test Data to NumPy Array
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
del y, train
# Check Fraud Proportion in Train and Validation Sets:
fraud_ratio_train = y_train.mean()  # Proportion in training set
fraud_ratio_val = y_val.mean()     # Proportion in validation set
print(f"Train fraud ratio: {fraud_ratio_train:.2%}")
print(f"Validation fraud ratio: {fraud_ratio_val:.2%}")

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)
del X_train, X_val, test

# Applying SMOTE to deal with the class imbalance by oversampling
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(X_train_smote.shape, y_train_smote.shape)
del X_train_scaled, y_train
pd.value_counts(y_train_smote)


# Convert NumPy arrays to DataFrames
X_train_smote_df = pd.DataFrame(X_train_smote)
y_train_smote_df = pd.DataFrame(y_train_smote, columns=['isFraud'])
X_val_scaled_df = pd.DataFrame(X_val_scaled)
y_val_df = pd.DataFrame(y_val, columns=['isFraud'])

# Save as CSV files
X_train_smote_df.to_csv('X_train_smote.csv', index=False)
y_train_smote_df.to_csv('y_train_smote.csv', index=False)
X_val_scaled_df.to_csv('X_val_scaled.csv', index=False)
y_val_df.to_csv('y_val.csv', index=False)