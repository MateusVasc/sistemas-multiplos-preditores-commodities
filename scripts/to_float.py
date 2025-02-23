import pandas as pd

def convert_y_to_float(df):
    """
    Converte a coluna 'y' para float, lidando com possíveis valores formatados como string com vírgula.
    
    :param df: DataFrame contendo a coluna 'y'.
    :return: DataFrame com 'y' convertido para float.
    """
    df = df.copy()
    df['y'] = df['y'].astype(str).str.replace(',', '.', regex=False)
    df['y'] = df['y'].astype(float)
    
    return df