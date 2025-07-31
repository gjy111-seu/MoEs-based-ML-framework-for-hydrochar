import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

df = pd.read_excel("Data file path", sheet_name='Cluster_4')
data = df[['C', 'H', 'O', 'N', 'FC', 'VM', 'A', 'T', 'RT', 'SL']]
target = df[['HHV', 'EY']]
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=66)


X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
Y_train_scaled = Y_scaler.fit_transform(Y_train)
Y_test_scaled = Y_scaler.transform(Y_test)

grid_params = {#'estimator__learning_rate': list(np.arange(0.01, 0.51, 0.01)),
               #'estimator__n_estimators': list(range(1, 201, 1)),
               #'estimator__max_depth': list(range(2, 11)),
               #'estimator__subsample': list(np.arange(0.1, 1.1, 0.1)),
               #'estimator__colsample_bytree': list(np.arange(0.1, 1.1, 0.1)),
               #'estimator__min_child_weight': list(range(2, 11))
               }

other_params = {'n_estimators': 101,
                'learning_rate': 0.16,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.6,
                'min_child_weight': 2,
                'random_state': 888,
                }

base_model = XGBRegressor(**other_params)

multi_xgb = MultiOutputRegressor(base_model)
optimized_param = GridSearchCV(estimator=multi_xgb, param_grid=grid_params, cv=5, scoring='r2', n_jobs=-1)
optimized_param.fit(X_train_scaled, Y_train_scaled)
chosen_param = optimized_param.best_params_
print("Best Parameters:", chosen_param)

best_gbr = optimized_param.best_estimator_

Y_pred_train_scaled = best_gbr.predict(X_train_scaled)
Y_pred_test_scaled = best_gbr.predict(X_test_scaled)


Y_pred_train = Y_scaler.inverse_transform(Y_pred_train_scaled)
Y_pred_test = Y_scaler.inverse_transform(Y_pred_test_scaled)



def evaluate_model(y_true, y_pred, dataset_name):
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)

    print(f"---- {dataset_name} results ----")
    print(f"HHV  - R²: {r2[0]:.4f}, RMSE: {rmse[0]:.4f}")
    print(f"EY   - R²: {r2[1]:.4f}, RMSE: {rmse[1]:.4f}")
    print("---------------------------------\n")


