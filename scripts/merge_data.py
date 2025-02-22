import pandas as pd

import sys
import os
sys.path.append(os.path.abspath("../"))

from scripts.save_to_parquet import save_to_parquet

def merge_parquet_dfs(dfs):
    """
    Junta uma lista de DataFrames do Pandas verticalmente.
    
    :param dfs: Lista de DataFrames a serem combinados.
    :return: DataFrame resultante da concatenação de todos os DataFrames da lista.
    """
    if not dfs:
        raise ValueError("A lista de DataFrames está vazia.")
    
    df_merged = pd.concat(dfs, axis=0, ignore_index=True)
    return df_merged

df_acucar_brl = pd.read_parquet('../data/acucar/acucar_brl.parquet')
df_algodao_brl = pd.read_parquet('../data/algodao/algodao_brl.parquet')
df_cafe_brl = pd.read_parquet('../data/cafe/cafe_brl.parquet')
df_milho_brl = pd.read_parquet('../data/milho/milho_brl.parquet')
df_soja_brl = pd.read_parquet('../data/soja/soja_brl.parquet')

df_acucar_usd = pd.read_parquet('../data/acucar/acucar_usd.parquet')
df_algodao_usd = pd.read_parquet('../data/algodao/algodao_usd.parquet')
df_cafe_usd = pd.read_parquet('../data/cafe/cafe_usd.parquet')
df_milho_usd = pd.read_parquet('../data/milho/milho_usd.parquet')
df_soja_usd = pd.read_parquet('../data/soja/soja_usd.parquet')

dfs_brl = [df_acucar_brl, df_algodao_brl, df_cafe_brl, df_milho_brl, df_soja_brl]
dfs_usd = [df_acucar_usd, df_algodao_usd, df_cafe_usd, df_milho_usd, df_soja_usd]

df_brl_final = merge_parquet_dfs(dfs_brl)
df_usd_final = merge_parquet_dfs(dfs_usd)

save_to_parquet(df_brl_final, "../data/all_comm/comm_brl.parquet")
save_to_parquet(df_usd_final, "../data/all_comm/comm_usd.parquet")