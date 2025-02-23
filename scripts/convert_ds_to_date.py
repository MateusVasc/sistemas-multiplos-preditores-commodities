import pandas as pd

def convert_ds_to_date(df):
    """
    Converte a coluna 'ds' para o formato de data (datetime).
    
    :param df: DataFrame contendo a coluna 'ds'.
    :return: DataFrame com 'ds' convertido para datetime.
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y', errors='coerce')
    return df