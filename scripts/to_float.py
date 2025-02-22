import pandas as pd

def convert_x_to_float(df):
    """
    Converte a coluna 'X' para float, lidando com possíveis valores formatados como string com vírgula.
    
    :param df: DataFrame contendo a coluna 'X'.
    :return: DataFrame com 'X' convertido para float.
    """
    df = df.copy()  # Evita modificar o original
    df['X'] = df['X'].astype(str).str.replace(',', '.', regex=False)  # Substituir vírgulas por pontos
    df['X'] = df['X'].astype(float)  # Converter para float
    
    return df