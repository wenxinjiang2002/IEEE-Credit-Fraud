"""
Created on Sun Oct 20 2024

EDA & Preprocessing

@author: Jiang
"""

# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
# ========================================================================
#                            Step 1: Import Datasets
# ========================================================================
print('---------------------------------------------------------')
print('Step 1: Import Datasets')
# ------------------------------------------------------------------------
#                                  merge & split
# ------------------------------------------------------------------------
starttime = datetime.datetime.now()
"""
全量数据集共有 
432个features 【40(identity)+392(transaction)】
1个outcome variable 【“isFraud”】
- 569877 Negative
- 20663 Positive

将该数据集根据TransactionID merge后，stratify split 分为 train & test（8:2）
"""
# Import datasets
# Step 1: 读取数据集
trans = pd.read_csv("train_transaction.csv")
ident = pd.read_csv("train_identity.csv")

# Step 2: 通过“TransactionID”合并两个数据集，并只保留一列“TransactionID”
merged_df = pd.merge(trans, ident, on='TransactionID')
#
# # Step 3: 使用train_test_split进行分层抽样，以确保“isFraud”比例相同
# train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42, stratify=merged_df['isFraud'])
#
# # Step 4: 保存训练集和测试集为 CSV 文件
# train_df.to_csv("train.csv", index=False)
# test_df.to_csv("test.csv", index=False)
#
# print("数据集已按'isFraud'比例拆分并保存为 'train.csv' 和 'test.csv'")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 7.8% Fraud, 92.2% Not Fraud

print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')


# ------------------------------------------------------------------------
#                            Step 2: Missing Value
# ------------------------------------------------------------------------
"""
Due to confidentiality issue, 一些variables的实际含义我们无从知晓，我想可能这些值缺失可能也有他的含义。
考虑到test data也会有很多null值，模型在训练时看到的 null 值越多，它就越能学习到如何处理这些缺失值。
在这种情况下，保留 null 值可以提高模型的泛化能力，使它在测试集上表现得更好。
 - 对于可以自然地处理缺失值的模型（比如一些树模型），使用把null值保留的数据集。
        - 直接使用"train.csv"
 - 其它不能自己处理null值的，则使用Imputation后的数据集。
"""
# ------------------------------------------------------------------------
#                  Review the percentage of missing value
# ------------------------------------------------------------------------
"""
After Review the percentage of missing value of each features.
Null Distribution are roughly shown as below:
100 %: 21 (features)
96 - 97%: 9
68 - 80%: 6
40 - 60%: 151
2 - 18%: 133
0 - 1%: 92
0: 22 (including outcome variable)
"""
# # Calculate the percentage of null values for each column in the train dataset
# null_percentage = train.isnull().sum() / len(train) * 100
#
# # Extract the data types of each column
# column_types = train.dtypes
#
# # Combine into a DataFrame
# null_percentage_df = pd.DataFrame({
#     'Column': train.columns,
#     'NullPercentage': null_percentage,
#     'VariableType': column_types
# })
#
# # Reset index to make it a proper DataFrame
# null_percentage_df.reset_index(drop=True, inplace=True)
#
# # Sort the DataFrame by NullPercentage in descending order
# null_percentage_df = null_percentage_df.sort_values(by='NullPercentage', ascending=False)
#
# # 输出一个missing value的预览
# null_percentage_df.to_csv('null_percentage_train.csv', index=False)

# null_percentage_train = pd.read_csv("null_percentage_train.csv")

# ------------------------------------------------------------------------
#                    handling missing value, 产出train_mod.csv
# ------------------------------------------------------------------------
'''
1. Features with over 96% missing values (30 features)
Method: drop these features

2. Features with 0 - 80% missing values (157 features)
Method: Impute missing values (“-” for categorical variables; median for numerical variables)
对于这一部分的categorical variables，我发现大多是设备型号这种。
我猜测如果是null的话，说明是这个公司无法检测到。我直觉认为这也是一个比较重要的feature。
(Possibly improve on:  identify 分布平均的 features 然后不要只fill median/mode。。。)

3. Features with 0 missing values (22 features)
Leave them Alone w Peace and Love.
'''
# # Step 1: 删除缺失值比例大于或等于96%的列
# columns_to_drop = null_percentage[null_percentage >= 96].index.tolist()
# train_mod = train.drop(columns=columns_to_drop)
#
# # Step 2: 分别处理剩下的分类变量和数值变量的缺失值
# # 填补分类变量的缺失值为 '-'
# categorical_columns = train_mod.select_dtypes(include=['object']).columns
# train_mod[categorical_columns] = train_mod[categorical_columns].fillna('-')
#
# # 填补数值变量的缺失值为中位数
# numerical_columns = train_mod.select_dtypes(include=['float64', 'int64']).columns
# train_mod[numerical_columns] = train_mod[numerical_columns].fillna(train_mod[numerical_columns].median())
#
# # Step 3: 再次检查missing value
# null_percentage_mod = train_mod.isnull().sum() / len(train_mod) * 100
# # Combine into a DataFrame
# null_percentage_mod_df = pd.DataFrame({
#     'Column': train_mod.columns,
#     'NullPercentage': null_percentage_mod,
# })
# # Reset index to make it a proper DataFrame
# null_percentage_mod_df.reset_index(drop=True, inplace=True)
# # Sort the DataFrame by NullPercentage in descending order
# null_percentage_mod_df = null_percentage_mod_df.sort_values(by='NullPercentage', ascending=False)
# null_percentage_mod_df.to_csv('null_percentage_train_mod.csv', index=False)
#
# # Step 4: output the modified dataset
# train_mod.to_csv('train_mod.csv', index=False)
# train_mod = pd.read_csv('train_mod.csv')

# 计算运行时间
endtime = datetime.datetime.now()
print('运行时间: ' + str(endtime - starttime))