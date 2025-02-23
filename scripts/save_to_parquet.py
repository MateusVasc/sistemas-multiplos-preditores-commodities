import pandas as pd
import os

def save_to_parquet(df, path_parquet):
    """
    Converte um DataFrame do Pandas para o formato Parquet e o salva no caminho especificado.

    :param df: DataFrame do Pandas que será convertido.
    :param path_parquet: Caminho completo do arquivo Parquet de saída, incluindo o nome do arquivo.
    :return: Nenhum retorno explícito, mas salva o arquivo convertido no local indicado.
    """
    dir_final = os.path.dirname(path_parquet)
    if not os.path.exists(dir_final):
        os.makedirs(dir_final)

    df.to_parquet(path_parquet, engine="pyarrow")
    print(f'Arquivo salvo em: {path_parquet}')