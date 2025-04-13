import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error , mean_squared_error, root_mean_squared_error, r2_score

def evaluate_forecasts(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'MAPE': mape,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
    }
