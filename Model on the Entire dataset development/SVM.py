import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor


df = pd.read_excel("Data file path", sheet_name='Sheet1')
data = df[['C', 'H', 'O', 'N', 'FC', 'VM', 'A', 'T', 'RT', 'SL']]
target = df[['HHV', 'EY']]


X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=42)


X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
Y_train_scaled = Y_scaler.fit_transform(Y_train)
Y_test_scaled = Y_scaler.transform(Y_test)

other_params = {'C': 1,
                'epsilon': 0.1,
                'kernel': 'rbf',
                'gamma': 'auto',
                }

base_svr = SVR(**other_params)


multi_svr = MultiOutputRegressor(base_svr)


param_grid = {
    #'estimator__C': [0.1, 1, 10, 100],
    #'estimator__epsilon': [0.01, 0.1, 0.2],
    #'estimator__kernel': ['linear', 'rbf', 'poly'],
    #'estimator__gamma': ['scale', 'auto']
}


grid_search = GridSearchCV(estimator=multi_svr, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, Y_train_scaled)


best_params = grid_search.best_params_
print("Best Parameters:", best_params)


best_svr = grid_search.best_estimator_


Y_pred_train_scaled = best_svr.predict(X_train_scaled)
Y_pred_test_scaled = best_svr.predict(X_test_scaled)

Y_pred_train = Y_scaler.inverse_transform(Y_pred_train_scaled)
Y_pred_test = Y_scaler.inverse_transform(Y_pred_test_scaled)


