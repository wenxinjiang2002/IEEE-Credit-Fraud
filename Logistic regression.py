import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
# Step 1: Prepare and standardize the Data
train_mod = pd.read_csv("train_mod.csv")
test = pd.read_csv("test.csv")

# Separate features and target variable
X_train = train_mod.drop(columns=['isFraud'])
X_test = test.drop(columns=['isFraud'])
y_train = train_mod['isFraud']
y_test = test['isFraud']

# 找到所有 object 类型的列
object_columns = X_train.select_dtypes(include=['object']).columns

# 删除 train 和 test 数据中的 object 列
X_train = X_train.drop(columns=object_columns)
X_test = X_test.drop(columns=object_columns)
# 对齐 train 和 test 数据集的列，保持两者的列一致
X_train, X_test = X_train.align(X_test, join='inner', axis=1)
# 选择数值型列，并将列名存储在 numerical_columns 中
numerical_columns = X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[numerical_columns] = X_test[numerical_columns].fillna(X_test[numerical_columns].median())

'''
在下面这段代码，每个 object 类型的列都会被遍历，转换为整数编码。转换后的数据只包含数值型列，不再有 object 类型的列。

# For each categorical column, apply LabelEncoder or get_dummies if there are multiple categories
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

注意事项
考虑到我们的object种类过多（电脑型号，）
如果 object 列有太多类别（如数千个独特值），可能会导致过拟合，（）可以考虑使用 One-Hot Encoding 或 降维。
（）LabelEncoder 不适合有序的分类数据（如 小、中、大），可以使用 OrdinalEncoder 来指定顺序关系。
'''
# Step 2: Apply SMOTE to address class imbalance in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 3: Standardize the data
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Step 4: Train the Logistic Regression Model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_resampled, y_train_resampled)

# Step 5: Make Predictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

from sklearn.metrics import confusion_matrix

# Calculate confusion matrix using actual labels (y_test) and predicted labels (y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate False Positive Rate
false_positive_rate = fp / (fp + tn)
print(f"False Positive Rate: {false_positive_rate:.4f}")