import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)
from typing import List, Dict, Callable, Optional
import numpy as np


class MetricRegistry:
    """Registry para métricas disponíveis"""

    METRICS = {
        'MAE': mean_absolute_error,
        'MAPE': mean_absolute_percentage_error,
        'MSE': mean_squared_error,
        'RMSE': root_mean_squared_error,
        'R2': r2_score
    }

    @classmethod
    def register_metric(cls, name: str, metric_func: Callable) -> None:
        cls.METRICS[name] = metric_func

    @classmethod
    def get_metric(cls, name: str) -> Callable:
        """Retorna função da métrica"""
        if name not in cls.METRICS:
            raise ValueError(f"Metric {name} não encontrada. Disponíveis: {list(cls.METRICS.keys())}")
        
        return cls.METRICS[name]
    
    @classmethod
    def list_metrics(cls) -> List:
        """Lista métricas disponíveis"""
        return list(cls.METRICS.keys())
    
class MetricEvaluator:
    """Avaliador de métricas para previsões de séries temporais"""

    def __init__(self, metrics: Optional[List[str]] = None) -> None:
        """
        Inicializa o avaliador

        Args:
            metrics: Lista de métricas para calcular. Se None, usa todas
        """
        self.metrics = metrics or MetricRegistry.list_metrics()
        self._validate_metrics()

    def _validate_metrics(self):
        """Valida se as métricas solicitadas existem"""
        available = MetricRegistry.list_metrics()
        invalid = [m for m in self.metrics if m not in available]
        if invalid:
            raise ValueError(f"Métricas inválidas: {invalid}. Disponíveis: {available}")
        
    def evaluate_single(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Avalia previsões para uma única série

        Args:
            y_true: Valores reais
            y_pred: Valores previstos
        """
        results = {}

        for metric_name in self.metrics:
            metric_func = MetricRegistry.get_metric(metric_name)

            try:
                results[metric_name] = metric_func(y_true, y_pred)
            except Exception as e:
                print(f"Erro ao calcular {metric_name}: {e}")
                results[metric_name] = np.nan

        return results
    
    def evaluate_multiple(
        self, 
        forecasts_df: pd.DataFrame, 
        actual_df: pd.DataFrame,
        model_columns: List[str],
        groupby_column: str = 'unique_id'
    ) -> pd.DataFrame:
        """
        Avalia previsões para múltiplas séries/modelos
        
        Args:
            forecasts_df: DataFrame com previsões
            actual_df: DataFrame com valores reais
            model_columns: Lista de colunas com previsões dos modelos
            groupby_column: Coluna para agrupar (ex: 'unique_id' para commodities)
            
        Returns:
            DataFrame com métricas por grupo e modelo
        """
        results = []
        
        # Merge dos dataframes
        merged_df = forecasts_df.merge(actual_df, on=['ds', groupby_column], how='inner')
        
        for group_name in merged_df[groupby_column].unique():
            group_data = merged_df[merged_df[groupby_column] == group_name]
            
            for model in model_columns:
                if model in group_data.columns:
                    # Remove NaN values
                    mask = ~(group_data['y'].isna() | group_data[model].isna())
                    y_true = group_data[mask]['y'].values
                    y_pred = group_data[mask][model].values
                    
                    if len(y_true) > 0:
                        metrics = self.evaluate_single(y_true, y_pred)
                        
                        # Criar novo dicionário
                        result_row = {
                            groupby_column: group_name,
                            'Model': model,
                            'n_observations': len(y_true),
                            **metrics  # Desempacotar as métricas
                        }
                        results.append(result_row)
        
        return pd.DataFrame(results)
    
    def evaluate_cross_validation(
        self,
        cv_results: pd.DataFrame,
        model_columns: List[str],
        groupby_column: str = 'unique_id'
    ) -> pd.DataFrame:
        """
        Avalia resultados de validação cruzada
        
        Args:
            cv_results: DataFrame com resultados de CV
            model_columns: Lista de colunas com previsões dos modelos
            groupby_column: Coluna para agrupar
            
        Returns:
            DataFrame com métricas por grupo, modelo e fold
        """
        results = []
        
        for group_name in cv_results[groupby_column].unique():
            group_data = cv_results[cv_results[groupby_column] == group_name]
            
            for cutoff in group_data['cutoff'].unique():
                cutoff_data = group_data[group_data['cutoff'] == cutoff]
                
                for model in model_columns:
                    if model in cutoff_data.columns:
                        mask = ~(cutoff_data['y'].isna() | cutoff_data[model].isna())
                        y_true = cutoff_data[mask]['y'].values
                        y_pred = cutoff_data[mask][model].values
                        
                        if len(y_true) > 0:
                            metrics = self.evaluate_single(y_true, y_pred)
                            
                            # Criar novo dicionário
                            result_row = {
                                groupby_column: group_name,
                                'Model': model,
                                'cutoff': cutoff,
                                'n_observations': len(y_true),
                                **metrics  # Desempacotar as métricas
                            }
                            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def add_metric(self, name: str, metric_func: Callable):
        """Adiciona uma nova métrica ao avaliador"""
        MetricRegistry.register_metric(name, metric_func)
        if name not in self.metrics:
            self.metrics.append(name)
    
    def get_summary_stats(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estatísticas resumo das métricas
        
        Args:
            metrics_df: DataFrame com métricas calculadas
            
        Returns:
            DataFrame com estatísticas resumo
        """
        numeric_cols = [col for col in metrics_df.columns if col in self.metrics]
        
        summary = metrics_df.groupby('Model')[numeric_cols].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(4)
        
        return summary