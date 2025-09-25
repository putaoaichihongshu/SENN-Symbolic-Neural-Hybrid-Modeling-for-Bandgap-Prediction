import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

features = ['FA', 'MA', 'Cs', 'I', 'Br', 'Cl']
target = 'Bg'
X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

train_metrics = {
    'R2': r2_score(y_train, y_train_pred),
    'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
    'MAE': mean_absolute_error(y_train, y_train_pred)
}
test_metrics = {
    'R2': r2_score(y_test, y_test_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
    'MAE': mean_absolute_error(y_test, y_test_pred)
}

print("Train set metrics:", train_metrics)
print("Test set metrics:", test_metrics)

df_pred = pd.DataFrame({
    'True_Bandgap': y_test,
    'Predicted_Bandgap': y_test_pred
})
df_pred.to_csv('lr_test_pred.csv', index=False)

with open('lr_metrics.txt', 'w') as f:
    f.write(f"Train set: {train_metrics}\n")
    f.write(f"Test set:  {test_metrics}\n")
