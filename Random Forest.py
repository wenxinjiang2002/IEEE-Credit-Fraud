"""
Created on FRI Nov 22 2024

Ensemble Methods: Bagging
 -- Random Forests --

@author: Jiang
"""

'''
Ensemble models are like a "team" of models working together to give better predictions. 
There are two main ways to build these teams: Bagging and Boosting. In this py, we will focus on bagging.
Bagging (e.g., Random Forest): 
    Each model in the team gets a random subset of the data and predicts independently. 
    Their outputs are then combined (e.g., by majority voting for classification tasks).

'''

# ========================================================================
#                            Step 1: Loading
# ========================================================================
print('---------------------------------------------------------')
print('Step 1: Loading')
# ------------------------------------------------------------------------
#                            import packages and data
# ------------------------------------------------------------------------
# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
from matplotlib import pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
final_train_df = pd.read_csv('CDA project/final_train_df.csv')
final_test_df = pd.read_csv('CDA project/final_test_df.csv')

# Split X and y
X_train = final_train_df.drop('isFraud', axis=1)
y_train = final_train_df['isFraud']
X_test = final_test_df.drop('isFraud', axis=1)
y_test = final_test_df['isFraud']

# 使用相同的随机森林模型
rfc = RandomForestClassifier(criterion='entropy', max_features='sqrt', max_samples=0.5, min_samples_split=80)
rfc.fit(X_train, y_train)
y_predproba = rfc.predict_proba(X_test)
print(f'Test AUC={roc_auc_score(y_test, y_predproba[:, 1])}')

# # 计算ROC曲线所需的数据
# from sklearn.metrics import roc_curve, auc, precision_recall_curve
# fpr, tpr, _ = roc_curve(y_test, y_predproba[:, 1])
# roc_auc = auc(fpr, tpr)

# # 计算PR曲线所需的数据
# precision, recall, _ = precision_recall_curve(y_test, y_predproba[:, 1])
# pr_auc = auc(recall, precision)

# # 创建子图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # 绘制ROC曲线
# ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
# ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# ax1.set_xlim([0.0, 1.0])
# ax1.set_ylim([0.0, 1.05])
# ax1.set_xlabel('False Positive Rate')
# ax1.set_ylabel('True Positive Rate')
# ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
# ax1.legend(loc="lower right")

# # 绘制PR曲线
# ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
# ax2.set_xlim([0.0, 1.0])
# ax2.set_ylim([0.0, 1.05])
# ax2.set_xlabel('Recall')
# ax2.set_ylabel('Precision')
# ax2.set_title('Precision-Recall Curve')
# ax2.legend(loc="lower left")

# plt.tight_layout()
# plt.savefig('curves.png')

# 特征重要性分析
plt.figure(figsize=(12, 8))  # 调整为单图大小

try:
    # 计算随机森林的特征重要性
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
    
    # 创建特征重要性DataFrame
    feature_importance_rf = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances,
        'std': std
    })
    feature_importance_rf = feature_importance_rf.sort_values('importance', ascending=False).head(20)
    
    # 绘制条形图（不使用xerr参数）
    ax = sns.barplot(
        x='importance', 
        y='feature', 
        data=feature_importance_rf, 
        palette='viridis'
    )
    
    # 手动添加误差条
    for i, row in feature_importance_rf.iterrows():
        ax.errorbar(
            x=row['importance'],
            y=i,
            xerr=row['std'],
            color='black',
            capsize=3,
            capthick=1,
            elinewidth=1,
            linestyle=''
        )
    
    # 添加标题和标签
    plt.title('Random Forest Feature Importance (Top 20)', pad=20, fontsize=12)
    plt.xlabel('Importance Score', fontsize=10)
    plt.ylabel('Features', fontsize=10)
    
    # 添加网格线便于阅读
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细的特征重要性报告
    print("\nRandom Forest Top 20 Important Features:")
    print("----------------------------------------")
    importance_report = feature_importance_rf.copy()
    importance_report['importance'] = importance_report['importance'].round(4)
    importance_report['std'] = importance_report['std'].round(4)
    print(importance_report.to_string(index=False))
    
    # 计算累积重要性
    cumulative_importance = np.cumsum(feature_importance_rf['importance'])
    print("\n累积重要性:")
    for i, cum_imp in enumerate(cumulative_importance, 1):
        print(f"Top {i} 特征累积重要性: {cum_imp:.4f}")

except Exception as e:
    print(f"生成特征重要性图时发生错误: {str(e)}")