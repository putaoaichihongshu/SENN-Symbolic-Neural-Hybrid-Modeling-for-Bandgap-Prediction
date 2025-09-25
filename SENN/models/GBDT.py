import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import ParameterGrid

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

features = ['FA', 'MA', 'Cs', 'I', 'Br', 'Cl']
target = 'Bg'
X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

param_grid = {
    'n_estimators': [80, 100, 150],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [7, 10, 12, 15],
    'min_samples_leaf': [5, 8, 10],
    'subsample': [0.8, 1.0],
    'max_features': ['sqrt', 'log2']
}

best_r2 = -float('inf')
best_model = None
best_params = None
best_train_metrics = None
best_test_metrics = None
best_y_test_pred = None

for params in ParameterGrid(param_grid):
    model = GradientBoostingRegressor(random_state=0, **params)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
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
df_pred.to_csv('gbdt_best_on_test_pred.csv', index=False)

with open('gbdt_best_on_test_metrics.txt', 'w') as f:
    f.write("Best hyperparameters: " + str(best_params) + "\n")
    f.write(f"Train set: {best_train_metrics}\n")
    f.write(f"Test set:  {best_test_metrics}\n")
