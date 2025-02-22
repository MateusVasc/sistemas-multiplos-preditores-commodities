import pandas as pd

def aggregate_monthly_avg(df):
    """
    Agrega a série temporal por mês, calculando a média dos valores dentro de cada mês.
    
    :param df: DataFrame contendo as colunas 'ds' (datas), 'X' (valores) e 'unique_id'.
    :return: Novo DataFrame com médias mensais, mantendo a data no formato 'AAAA-MM-01'.
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])

    df['year_month'] = df['ds'].dt.to_period('M')

    df_monthly = df.groupby(['year_month', 'unique_id']).agg({'X': 'mean'}).reset_index()

    df_monthly['ds'] = df_monthly['year_month'].astype(str) + '-01'
    df_monthly['ds'] = pd.to_datetime(df_monthly['ds'])

    df_monthly = df_monthly[['ds', 'X', 'unique_id']]
    
    return df_monthly