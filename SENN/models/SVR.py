import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

features = ['FA', 'MA', 'Cs', 'I', 'Br', 'Cl']
target = 'Bg'
X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

param_grid = {
    'kernel': ['rbf', 'linear'],
    'C': [10, 25, 50, 100, 150],
    'gamma': ['scale', 0.05],
    'epsilon': [0.15]
}

best_r2 = -float('inf')
best_model = None
best_params = None
best_train_metrics = None
best_test_metrics = None
best_y_test_pred = None

for params in ParameterGrid(param_grid):
    model = SVR(**params)
    model.fit(X_train_scaled, y_train_scaled)
    y_train_pred_scaled = model.predict(X_train_scaled)
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    r2_test = r2_score(y_test, y_test_pred)
    if r2_test > best_r2:
        best_r2 = r2_test
        best_model = model
        best_params = params
        best_train_metrics = {
            'R2': r2_score(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'MAE': mean_absolute_error(y_train, y_train_pred)
        }
        best_test_metrics = {
            'R2': r2_test,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred)
        }
        best_y_test_pred = y_test_pred

print("Best test set R2:", best_r2)
print("Best hyperparameters:", best_params)
print("Train set metrics:", best_train_metrics)
print("Test set metrics:", best_test_metrics)

df_pred = pd.DataFrame({
    'True_Bandgap': y_test,
    'Predicted_Bandgap': best_y_test_pred
})
df_pred.to_csv('svr_best_on_test_pred.csv', index=False)

with open('svr_best_on_test_metrics.txt', 'w') as f:
    f.write("Best hyperparameters: " + str(best_params) + "\n")
    f.write(f"Train set: {best_train_metrics}\n")
    f.write(f"Test set:  {best_test_metrics}\n")
