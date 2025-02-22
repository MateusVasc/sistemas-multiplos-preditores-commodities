import pandas as pd

def rename_and_add_id(df, cols_dict, unique_id):
    """
    Renomeia colunas de um DataFrame e adiciona uma nova coluna chamada 'unique_id'.
    
    :param df: DataFrame original.
    :param cols_dict: Dicionário onde a chave é o nome antigo da coluna e o valor é o novo nome.
    :param unique_id: Valor que será atribuído à nova coluna 'unique_id'.
    :return: Novo DataFrame com colunas renomeadas e a coluna 'unique_id' adicionada.
    """
    df = df.rename(columns=cols_dict, inplace=False)
    df['unique_id'] = unique_id
    return df