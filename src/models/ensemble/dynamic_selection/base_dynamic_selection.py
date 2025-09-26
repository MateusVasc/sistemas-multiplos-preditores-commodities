from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DynamicSelection(ABC):
    def __init__(self, base_models, windows_size=12):
        self.base_models = base_models
        self.windows_size = windows_size
        self.history_errors = {model: [] for model in base_models}

    def _extract_lag_windows(self, series):
        """Cria janelas de lags e targets a partir de uma série univariada."""
        windows, targets = [], []
        for i in range(self.windows_size, len(series)-1):
            windows.append(series[i-self.windows_size:i])
            targets.append(series[i])
        return np.array(windows), np.array(targets)

    def fit(self, X, y):
        """Treina cada modelo base uma vez no conjunto global."""
        for model in self.base_models:
            model.fit(X, y)

    def update_history(self, y_true, preds_dict):
        """Atualiza erros históricos para cada modelo base."""
        for model, y_pred in preds_dict.items():
            error = abs(y_true - y_pred)
            self.history_errors[model].append(error)

    @abstractmethod
    def predict(self, y_series, horizon=1):
        """Cada subclasse deve implementar sua lógica de previsão."""
        pass