from src.models.ensemble.dynamic_selection.base_dynamic_selection import DynamicSelection

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DSNAW(DynamicSelection):
    def __init__(self, base_models, last_k, windows_size=12):
        """
        Dynamic Classifier Selection by Local Accuracy para séries temporais.

        base_models : list
            Lista de modelos de regressão (base class).
        window_size : int
            Quantidade de lags para criar as janelas (base class).
        top_k : int
            Número de janelas utilizadas na formulação da RoC.
        """
        super().__init__(base_models, windows_size)
        self.last_k = last_k

    def _evaluate_models(self, X, y):
        """Treina cada modelo nos dados de competência e retorna os erros (MAE)."""
        errors = []
        for model in self.base_models:
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            errors.append(mae)
        return np.array(errors)

    def predict(self, y_series, horizon): # type: ignore
        """
        Faz previsão recursiva de 'horizon' passos à frente.

        y_series : array-like
            Série temporal completa.
        horizon : int
            Quantidade de passos futuros a prever.
        """
        # to array numpy se necessário
        if hasattr(y_series, 'values'):  # pandas Series
            y_series = y_series.values
        elif not isinstance(y_series, np.ndarray):  # outro tipo
            y_series = np.array(y_series)
        
        # Verificar se há dados suficientes
        if len(y_series) < self.windows_size + horizon:
            raise ValueError(f"Série temporal muito curta. Necessário pelo menos {self.windows_size + horizon} pontos, mas recebido {len(y_series)}")
        
        y_train = y_series[:-horizon]
        y_real_future = y_series[-horizon:]
        curr_lags = y_train[-self.windows_size:]
        forecast = []

        for step in range(horizon):
            windows, targets = self._extract_lag_windows(y_train)

            competence_X = windows[:-self.last_k]
            competence_y = targets[:-self.last_k]

            # pegar melhor modelo
            errors = self._evaluate_models(competence_X, competence_y)
            best_model = self.base_models[np.argmin(errors)]
            print(best_model)

            # prever próximo passo e atualizar histórico de erros
            y_next = best_model.predict([curr_lags])[0]
            self.update_history(
                y_real_future[step],
                {best_model: y_next}
            )
            forecast.append(y_next)

            # atualizar histórico da série
            curr_lags = np.append(curr_lags[1:], y_real_future[step])
            y_train = np.append(y_train, y_real_future[step])

        return np.array(forecast), y_real_future