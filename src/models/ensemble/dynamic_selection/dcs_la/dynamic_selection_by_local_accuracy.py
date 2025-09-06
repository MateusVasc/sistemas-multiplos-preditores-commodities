from src.models.ensemble.dynamic_selection.base_dynamic_selection import DynamicSelection

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import mean_squared_error

class DCSLARegressor(DynamicSelection):
    def __init__(self, base_models, windows_size=12, top_k=10, similarity="cosine"):
        """
        Dynamic Classifier Selection by Local Accuracy para séries temporais.

        base_models : list
            Lista de modelos de regressão (Vem da base class).
        window_size : int
            Quantidade de lags para criar as janelas (Vem da base class).
        top_k : int
            Número de vizinhos mais similares usados para avaliar competência.
        similarity : str
            Métrica de similaridade: 'cosine' ou 'euclidean'.
        """
        super().__init__(base_models, windows_size)
        self.top_k = top_k
        self.similarity = similarity

    def _evaluate_models(self, X, y):
        """Treina cada modelo nos dados de competência e retorna os erros (MSE)."""
        errors = []
        for model in self.base_models:
            model.fit(X, y)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            errors.append(mse)
        return np.array(errors)

    def predict(self, y_series, horizon=1): # type: ignore
        """
        Faz previsão recursiva de 'horizon' passos à frente.

        y_series : array-like
            Série temporal completa.
        horizon : int
            Quantidade de passos futuros a prever.
        """
        # Converter para array numpy se necessário
        if hasattr(y_series, 'values'):  # pandas Series
            y_series = y_series.values
        elif not isinstance(y_series, np.ndarray):  # lista ou outro tipo
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

            # calcular similaridade/distância
            if self.similarity == "cosine":
                sims = cosine_similarity([curr_lags], windows)[0] # type: ignore
                top_k_idx = sims.argsort()[-self.top_k:]
            else:  # euclidean
                dists = euclidean_distances([curr_lags], windows)[0] # type: ignore
                top_k_idx = dists.argsort()[:self.top_k]

            competence_X = windows[top_k_idx]
            competence_y = targets[top_k_idx]

            # escolher melhor modelo
            errors = self._evaluate_models(competence_X, competence_y)
            best_model = self.base_models[np.argmin(errors)]

            # prever próximo passo
            y_next = best_model.predict([curr_lags])[0]
            forecast.append(y_next)

            # atualizar histórico
            curr_lags = np.append(curr_lags[1:], y_real_future[step])
            y_train = np.append(y_train, y_real_future[step])

        return np.array(forecast), y_real_future