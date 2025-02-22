import pandas as pd
import matplotlib.pyplot as plt

def plot_time_series(df, value_column):
    """
    Plota um gráfico de séries temporais com 'ds' no eixo X e a coluna especificada no eixo Y.
    
    :param df: DataFrame contendo a coluna 'ds' e a coluna de valores.
    :param value_column: Nome da coluna que será usada como eixo Y.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df[value_column], linestyle='-')
    plt.xlabel('Data')
    plt.ylabel(value_column)
    plt.title(f'Série Temporal de {value_column}')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()