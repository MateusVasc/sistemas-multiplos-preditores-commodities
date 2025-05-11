import os
import pandas as pd

def split_currency_data(df):
    """
    Separa um DataFrame com colunas de moedas BRL e USD em dois DataFrames distintos.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'Data', 'À vista R$', e 'À vista US$'.

    Retorna:
    --------
    tuple
        Dois DataFrames: um com os dados em reais e outro em dólares.
    """
    df_brl = df[['Data', 'À vista R$']].copy()
    df_usd = df[['Data', 'À vista US$']].copy()
    return df_brl, df_usd


def extract_currency_series(df, currency="BRL"):
    """
    Extrai a série temporal da moeda especificada (BRL ou USD).

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'Data', 'À vista R$', e 'À vista US$'.

    currency : str
        Tipo de moeda a ser extraída: 'BRL' ou 'USD'.

    Retorna:
    --------
    pd.DataFrame
        DataFrame com colunas 'Data' e o valor da moeda selecionada.

    Exceções:
    ---------
    AttributeError
        Se o tipo de moeda fornecido for inválido.
    """
    if currency == "BRL":
        return df[['Data', 'À vista R$']].copy()
    elif currency == "USD":
        return df[['Data', 'À vista US$']].copy()
    else:
        raise AttributeError("Invalid currency type")
    

def extract_currency_series_algodao(df, currency="BRL"):
    if currency == "BRL":
        return df[['Data', 'Prazo de 8 dias R$']].copy()
    elif currency == "USD":
        return df[['Data', 'Prazo de 8 dias US$']].copy()
    else:
        raise AttributeError("Invalid currency type")


def rename_columns_and_set_id(df, cols_dict, unique_id):
    """
    Renomeia colunas e adiciona um identificador único ao DataFrame.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame original.

    cols_dict : dict
        Dicionário de mapeamento para renomear colunas. Exemplo: {'Data': 'ds', 'Valor': 'y'}

    unique_id : str
        Identificador a ser adicionado a todas as linhas do DataFrame.

    Retorna:
    --------
    pd.DataFrame
        DataFrame com colunas renomeadas e nova coluna 'unique_id'.
    """
    df = df.rename(columns=cols_dict, inplace=False)
    df['unique_id'] = unique_id
    return df


def convert_column_to_datetime(df, column):
    """
    Converte a coluna alvo (column) de strings para objetos datetime.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo a coluna alvo com datas em formato string.

    column : str
        Coluna alvo do dataframe que precisa e será convertida.

    Retorna:
    --------
    pd.DataFrame
        DataFrame com a coluna convertida para datetime.
    """
    df = df.copy()
    df[column] = pd.to_datetime(df[column], format='%d/%m/%Y', errors='coerce')
    return df


def convert_column_to_float(df, column):
    """
    Converte a coluna alvo (column) de string para float, tratando vírgulas como separadores decimais.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo a coluna alvo com valores numéricos como strings.

    column : str
        Coluna alvo do dataframe que precisa e será convertida.

    Retorna:
    --------
    pd.DataFrame
        DataFrame com a coluna alvo convertida para tipo float.
    """
    df = df.copy()
    df[column] = df[column].astype(str).str.replace(',', '.', regex=False)
    df[column] = df[column].astype(float)
    
    return df


def aggregate_monthly_mean(df):
    """
    Agrega a série temporal calculando a média mensal por identificador único.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo colunas 'ds' (datetime), 'y' (valor), e 'unique_id'.

    Retorna:
    --------
    pd.DataFrame
        DataFrame com a média mensal por 'unique_id' no formato adequado para modelagem temporal.
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['year_month'] = df['ds'].dt.to_period('M')

    df_monthly = df.groupby(['year_month', 'unique_id']).agg({'y': 'mean'}).reset_index()
    df_monthly['ds'] = pd.to_datetime(df_monthly['year_month'].astype(str) + '-01')
    df_monthly = df_monthly[['ds', 'y', 'unique_id']]
    
    return df_monthly


def drop_over_limit_date(df, limit_date, column_date):
    """
    Remove as linhas do DataFrame cuja data em uma coluna especificada excede uma data limite.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo uma coluna de datas.

    limit_date : str ou datetime
        Data limite no formato reconhecido pelo pandas (ex: '2023-01-01'). 
        Linhas com datas posteriores a essa serão removidas.

    column_date : str
        Nome da coluna no DataFrame que contém os valores de data a serem comparados.

    Retorna:
    --------
    pd.DataFrame
        DataFrame filtrado com apenas as linhas anteriores à data limite.
    """
    limit_date = pd.to_datetime(limit_date, format='%d/%m/%Y')
    df = df[df[column_date] < limit_date]
    return df


def export_to_parquet(df, path_parquet):
    """
    Salva um DataFrame em formato Parquet no caminho especificado.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame a ser salvo.

    path_parquet : str
        Caminho completo para salvar o arquivo .parquet.

    Retorna:
    --------
    None
        Salva o arquivo e imprime uma mensagem de sucesso.
    """
    dir_final = os.path.dirname(path_parquet)
    if not os.path.exists(dir_final):
        os.makedirs(dir_final)

    df.to_parquet(path_parquet, engine="pyarrow")
    print(f'Arquivo salvo em: {path_parquet}')
