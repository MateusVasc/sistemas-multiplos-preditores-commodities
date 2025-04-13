import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    metrics: List[str] = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']
) -> None:
    """
    Plota comparação de métricas entre modelos.
    
    Args:
        metrics_df: DataFrame com métricas calculadas
        metrics: Lista de métricas a serem plotadas
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        sns.barplot(
            data=metrics_df,
            x='Commodity',
            y=metric,
            hue='Model',
            ax=ax
        )
        ax.set_title(f'{metric} por Commodity')
        ax.tick_params(axis='x', rotation=45)
    
    fig.delaxes(axes[2, 1])
    
    plt.tight_layout()
    plt.show()


def plot_forecasts_grid(
    actual: pd.DataFrame,
    forecasts: pd.DataFrame,
    models: List[str],
    commodities: List[str],
    n_cols: int = 2
) -> None:
    """
    Plota grid de previsões para múltiplas commodities.
    
    Args:
        actual: DataFrame com valores reais
        forecasts: DataFrame com previsões
        models: Lista de nomes dos modelos
        commodities: Lista de commodities
        n_cols: Número de colunas no grid
    """
    n_rows = (len(commodities) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    for idx, commodity in enumerate(commodities):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        ax.plot(
            actual[actual['unique_id'] == commodity]['ds'],
            actual[actual['unique_id'] == commodity]['y'],
            label='Valor Real',
            color='black'
        )
        
        for model in models:
            ax.plot(
                forecasts[forecasts['unique_id'] == commodity]['ds'],
                forecasts[forecasts['unique_id'] == commodity][model],
                label=model
            )
        
        ax.set_title(commodity)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(models)+1)
    plt.show()


def plot_validation_forecasts(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    forecasts_val: pd.DataFrame,
    commodity: str,
    models: List[str] = ['Naive', 'AutoARIMA']
) -> None:
    """
    Plota previsões de validação vs valores reais para uma commodity.
    
    Args:
        train_data: DataFrame com dados de treino
        val_data: DataFrame com dados de validação
        forecasts_val: DataFrame com previsões de validação
        commodity: Nome da commodity
        models: Lista de modelos a serem plotados
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(
        train_data[train_data['unique_id'] == commodity]['ds'],
        train_data[train_data['unique_id'] == commodity]['y'],
        label=f'Real {commodity} (Treino)',
        linestyle='--',
        color='gray'
    )
    
    plt.plot(
        val_data[val_data['unique_id'] == commodity]['ds'],
        val_data[val_data['unique_id'] == commodity]['y'],
        label=f'Real {commodity} (Validação)',
        linestyle='--',
        color='black'
    )
    
    for model in models:
        plt.plot(
            forecasts_val[forecasts_val['unique_id'] == commodity]['ds'],
            forecasts_val[forecasts_val['unique_id'] == commodity][model],
            label=f'Previsão {model}'
        )
    
    plt.legend()
    plt.title(f'Previsões na Validação vs Valores Reais - {commodity}')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_test_forecasts(
    full_train: pd.DataFrame,
    test_data: pd.DataFrame,
    forecasts_test: pd.DataFrame,
    commodity: str,
    models: List[str] = ['Naive', 'AutoARIMA']
) -> None:
    """
    Plota previsões de teste vs valores reais para uma commodity.
    
    Args:
        full_train: DataFrame com dados de treino + validação
        test_data: DataFrame com dados de teste
        forecasts_test: DataFrame com previsões de teste
        commodity: Nome da commodity
        models: Lista de modelos a serem plotados
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(
        full_train[full_train['unique_id'] == commodity]['ds'],
        full_train[full_train['unique_id'] == commodity]['y'],
        label=f'Real {commodity} (Treino + Validação)',
        linestyle='--',
        color='gray'
    )
    
    plt.plot(
        test_data[test_data['unique_id'] == commodity]['ds'],
        test_data[test_data['unique_id'] == commodity]['y'],
        label=f'Real {commodity} (Teste)',
        linestyle='--',
        color='black'
    )
    
    for model in models:
        plt.plot(
            forecasts_test[forecasts_test['unique_id'] == commodity]['ds'],
            forecasts_test[forecasts_test['unique_id'] == commodity][model],
            label=f'Previsão {model}'
        )
    
    plt.legend()
    plt.title(f'Previsões no Teste vs Valores Reais - {commodity}')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
