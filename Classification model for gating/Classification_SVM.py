import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder
import numpy as np
from scipy import interpolate


df = pd.read_excel("Data file path", sheet_name='Sheet1')
X = df[['C', 'H', 'O', 'N', 'FC', 'VM', 'A', 'T', 'RT', 'SL']]
y = df['Cluster']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'poly', 'linear'],
    'gamma': ['scale', 'auto']
}

svm_clf = SVC(decision_function_shape='ovr', probability=True)

grid_search = GridSearchCV(
    estimator=svm_clf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test_scaled)
Y_score = best_model.predict_proba(X_test_scaled)


n_classes = len(np.unique(y_test))
Y_test_bin = label_binarize(y_test, classes=np.unique(y_test))


precision = dict()
recall = dict()
average_precision = dict()
thresholds = dict()

for i in range(n_classes):
    precision[i], recall[i], thresholds[i] = precision_recall_curve(
        Y_test_bin[:, i],
        Y_score[:, i]
    )
    average_precision[i] = average_precision_score(Y_test_bin[:, i], Y_score[:, i])


precision["micro"], recall["micro"], _ = precision_recall_curve(
    Y_test_bin.ravel(),
    Y_score.ravel()
)
average_precision["micro"] = average_precision_score(Y_test_bin, Y_score, average="micro")


pr_data = pd.DataFrame()


for i in range(n_classes):
    temp_df = pd.DataFrame({
        'Class': i,
        'Precision': precision[i],
        'Recall': recall[i],
        'Threshold': np.append(thresholds[i], np.nan)
    })
    pr_data = pd.concat([pr_data, temp_df], ignore_index=True)

# 保存微平均数据
micro_df = pd.DataFrame({
    'Class': 'micro',
    'Precision': precision["micro"],
    'Recall': recall["micro"],
    'Threshold': np.nan  # 微平均无阈值
})
pr_data = pd.concat([pr_data, micro_df], ignore_index=True)

# 保存到Excel
excel_path = "save path"
pr_data.to_excel(excel_path, index=False)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
class_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

