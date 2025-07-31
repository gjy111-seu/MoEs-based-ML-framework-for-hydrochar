import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import interpolate


df = pd.read_excel("Data file path", sheet_name='Sheet1')
data = df[['C', 'H', 'O', 'N', 'FC', 'VM', 'A', 'T', 'RT', 'SL']]
target = df[['Cluster']]
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=42, stratify=target)

grid_params = {#'learning_rate': list(np.arange(0.01, 0.51, 0.01)),
               #'n_estimators': list(range(300, 401, 1)),
               #'max_depth': list(range(2, 11)),
               #'subsample': list(np.arange(0.1, 1.1, 0.1)),
               #'colsample_bytree': list(np.arange(0.1, 1.1, 0.1)),
               #'min_child_weight': list(range(2, 11))
               }

other_params = {'n_estimators': 307,
                'learning_rate': 0.22,
                'max_depth': 4,
                'subsample': 0.9,
                'colsample_bytree': 0.3,
                'min_child_weight': 9,
                'random_state': 888,
                }

xgb = XGBClassifier(**other_params, eval_metric='mlogloss')

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=grid_params,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, np.ravel(Y_train))
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)
Y_score = best_model.predict_proba(X_test)  # 获取预测概率


n_classes = len(np.unique(Y_test))
Y_test_bin = label_binarize(Y_test, classes=np.unique(Y_test))


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
        'Threshold': np.append(thresholds[i], np.nan)  # 阈值比P-R少1个
    })
    pr_data = pd.concat([pr_data, temp_df], ignore_index=True)


micro_df = pd.DataFrame({
    'Class': 'micro',
    'Precision': precision["micro"],
    'Recall': recall["micro"],
    'Threshold': np.nan  # 微平均无阈值
})
pr_data = pd.concat([pr_data, micro_df], ignore_index=True)


excel_path = "save path"
pr_data.to_excel(excel_path, index=False)


grid_search.fit(X_train, Y_train)


print("Best parameters found: ", grid_search.best_params_)
Y_pred = grid_search.predict(X_test)


print("Accuracy:", accuracy_score(Y_test, Y_pred))


print("Classification Report:")
print(classification_report(Y_test, Y_pred, digits=4))


cm = confusion_matrix(Y_test, Y_pred)
conf_matrix = pd.DataFrame(cm, index=['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4'], columns=['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4'])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


