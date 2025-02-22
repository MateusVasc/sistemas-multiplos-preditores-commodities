import pandas as pd
import os

def csv_to_parquet(path_csv, path_final="../data"):
    """
    Converte um arquivo CSV para o formato Parquet e o salva no diretório especificado.
    
    :param path_csv: Caminho do arquivo CSV de entrada.
    :param path_final: Diretório onde o arquivo Parquet será salvo (padrão: "../data").
    :return: Nenhum retorno explícito, mas salva o arquivo convertido no local indicado.
    """
    df = pd.read_csv(path_csv, sep=';', on_bad_lines='warn')

    if not os.path.exists(path_final):
        os.makedirs(path_final)

    path_arquivo_parquet = os.path.join(path_final, os.path.splitext(os.path.basename(path_csv))[0] + ".parquet")
    
    df.to_parquet(path_arquivo_parquet, engine="pyarrow")
    print(f'Arquivo salvo em: {path_arquivo_parquet}')


csv_to_parquet("../data/Dados_commodities_mensal_nova.csv")