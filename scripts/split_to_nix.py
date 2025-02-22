import pandas as pd

def split_currency_columns(df):
    """
    Separa um DataFrame em duas cópias, uma com a coluna 'À vista R$' e outra com a coluna 'À vista US$'.
    
    :param df: DataFrame original com as colunas 'Data', 'À vista R$', e 'À vista US$'.
    :return: Duas cópias do DataFrame, uma para cada moeda.
    """
    df_brl = df[['Data', 'À vista R$']].copy()
    df_usd = df[['Data', 'À vista US$']].copy()
    return df_brl, df_usd