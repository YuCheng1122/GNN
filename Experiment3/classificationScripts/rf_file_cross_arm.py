import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# 載入訓練資料（ARM + X86-64）
train_df = pd.read_csv('/home/tommy/Projects/cross-architecture/Experiment3.1/dataset/cleaned_20250509_train_450.csv')
print(f"訓練資料集（ARM + X86-64）樣本數: {len(train_df)}")

# 載入測試資料（MIPS）
mips_df = pd.read_csv('/home/tommy/Projects/cross-architecture/Experiment3.1/dataset/cleaned_20250509_test_600.csv')
print(f"測試資料集（MIPS）樣本數: {len(mips_df)}")

# 檢查標籤分布
for df, name in [(train_df, "ARM + X86-64"), (mips_df, "MIPS")]:
    label_counts = df['label'].value_counts()
    print(f"\n{name} 標籤分布:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} samples")

# 特徵欄位設定
excluded_columns = ['file_name', 'CPU', 'label']
feature_columns = [col for col in train_df.columns if col not in excluded_columns]

# 確保特徵欄一致
common_features = [col for col in feature_columns if col in mips_df.columns]
if len(common_features) != len(feature_columns):
    print(f"\n警告：MIPS 測試集中缺少部分特徵。將使用共同特徵欄 {len(common_features)} 項。")
    feature_columns = common_features

print(f"\n使用的特徵數量: {len(feature_columns)}")

# 建立結果輸出資料夾
results_dir = 'cross_arch_train_450_test_mips_results'
os.makedirs(results_dir, exist_ok=True)

# 建立訓練與測試資料
X_train = train_df[feature_columns].values
y_train = train_df['label'].values
X_test = mips_df[feature_columns].values
y_test = mips_df['label'].values

# 訓練模型
print("\n訓練 Random Forest 模型（使用 ARM + X86-64 資料）...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 預測
print("使用模型對 MIPS 資料進行預測...")
y_pred = rf.predict(X_test)

# 評估指標
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nMIPS 測試資料之模型表現：")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")

# 詳細報告
class_report = classification_report(y_test, y_pred)
print("\n分類報告:")
print(class_report)
with open(f'{results_dir}/classification_report.txt', 'w') as f:
    f.write(class_report)

# 預測結果另存
mips_df['predicted_label'] = y_pred
mips_df.to_csv(f'{results_dir}/mips_test_predictions.csv', index=False)

# 混淆矩陣圖
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=sorted(mips_df['label'].unique()),
            yticklabels=sorted(mips_df['label'].unique()),
            cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on MIPS Test Data')
plt.tight_layout()
plt.savefig(f'{results_dir}/confusion_matrix.png')
plt.close()

# 特徵重要性
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
print("\nTop 10 最重要的特徵：")
for i in range(min(10, len(feature_columns))):
    idx = sorted_idx[i]
    print(f"{feature_columns[idx]}: {feature_importance[idx]:.4f}")

# 畫出特徵重要性圖
plt.figure(figsize=(12, 8))
top_n = 20
plt.barh(range(top_n), feature_importance[sorted_idx[:top_n]])
plt.yticks(range(top_n), [feature_columns[i] for i in sorted_idx[:top_n]])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig(f'{results_dir}/feature_importance.png')
plt.close()

# 存成總結報告
with open(f'{results_dir}/performance_metrics_summary.txt', 'w') as f:
    f.write("Cross-Architecture ML 評估結果\n")
    f.write("=====================================\n\n")
    f.write("訓練資料: ARM + X86-64\n")
    f.write(f"樣本數: {len(train_df)}\n")
    f.write("測試資料: MIPS\n")
    f.write(f"樣本數: {len(mips_df)}\n\n")
    f.write(f"使用特徵數: {len(feature_columns)}\n\n")

    f.write("模型表現:\n")
    f.write(f"  Accuracy: {accuracy:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall: {recall:.4f}\n")
    f.write(f"  F1 Score: {f1:.4f}\n\n")

    f.write("分類報告:\n")
    f.write(class_report)
    f.write("\n混淆矩陣:\n")
    f.write(str(conf_matrix))
    f.write("\n\nTop 10 最重要特徵:\n")
    for i in range(min(10, len(feature_columns))):
        idx = sorted_idx[i]
        f.write(f"{feature_columns[idx]}: {feature_importance[idx]:.4f}\n")

print(f"\n所有結果已儲存於資料夾：'{results_dir}'")
print("Summary: 本次實驗測試模型在 ARM + X86-64 訓練下，對 MIPS 架構的分類表現。")
