import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def plot_raw_series(df, value_column):
    """
    Plota uma série temporal com base em um DataFrame e uma coluna de valores especificada.

    A função gera um gráfico de linha com a coluna de datas ('ds') no eixo x e os valores da 
    coluna especificada no eixo y. É útil para visualizar tendências temporais de variáveis.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo a série temporal. Deve conter uma coluna 'ds' com datas e uma 
        coluna com os valores a serem plotados.

    value_column : str
        Nome da coluna do DataFrame que contém os valores a serem utilizados no eixo y do gráfico.

    Retorna:
    --------
    None
        A função apenas exibe o gráfico e não retorna nenhum valor.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df[value_column], linestyle='-')
    plt.xlabel('Data')
    plt.ylabel(value_column)
    plt.title(f'Série Temporal de {value_column}')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def plot_series_acf(series):
    """
    Plota o gráfico da Função de Autocorrelação (ACF) para uma série temporal.

    Parâmetros:
    -----------
    series : pd.Series ou np.ndarray
        Série temporal a ser analisada.

    Retorna:
    --------
    None
        Apenas exibe o gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(series, lags=100, ax=ax)
    ax.set_title("ACF da Série Temporal")
    plt.show()

def plot_series_pacf(series):
    """
    Plota o gráfico da Função de Autocorrelação Parcial (PACF) para uma série temporal.

    Parâmetros:
    -----------
    series : pd.Series ou np.ndarray
        Série temporal a ser analisada.

    Retorna:
    --------
    None
        Apenas exibe o gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_pacf(series, lags=100, ax=ax)
    ax.set_title("PACF da Série Temporal")
    plt.show()

def checks_stationarity(series):
    """
    Realiza o teste de Dickey-Fuller Aumentado (ADF) para verificar a estacionariedade da série temporal.

    O teste avalia a hipótese nula de que a série possui raiz unitária (não é estacionária).

    Parâmetros:
    -----------
    series : pd.Series ou np.ndarray
        Série temporal a ser testada.

    Retorna:
    --------
    None
        Imprime o resultado do teste e se a série pode ser considerada estacionária.
    """
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Críticos:", result[4])

    if result[1] > 0.05:
        print("❌ Série não é estacionária.")
    else:
        print("✅ Série é estacionária.")