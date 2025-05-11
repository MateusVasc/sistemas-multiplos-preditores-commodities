import pandas as pd

import sys
import os
sys.path.append(os.path.abspath("../"))

from scripts.time_series_preprocessing import export_to_parquet

def merge_parquet_dfs(dfs):
    if not dfs:
        raise ValueError("A lista de DataFrames está vazia.")
    
    df_merged = pd.concat(dfs, axis=0, ignore_index=True)
    return df_merged

df_acucar_santos_brl = pd.read_parquet('../data/acucar/acucar_santos_brl.parquet')
df_acucar_sp_brl = pd.read_parquet('../data/acucar/acucar_sp_brl.parquet')
df_algodao_brl = pd.read_parquet('../data/algodao/algodao_brl.parquet')
df_arroz_brl = pd.read_parquet('../data/arroz/arroz_brl.parquet')
df_cafe_arabica_brl = pd.read_parquet('../data/cafe/cafe_arabica_brl.parquet')
df_cafe_robusta_brl = pd.read_parquet('../data/cafe/cafe_robusta_brl.parquet')
df_milho_brl = pd.read_parquet('../data/milho/milho_brl.parquet')
df_soja_parana_brl = pd.read_parquet('../data/soja/soja_parana_brl.parquet')
df_soja_paranagua_brl = pd.read_parquet('../data/soja/soja_paranagua_brl.parquet')
df_trigo_parana_brl = pd.read_parquet('../data/trigo/trigo_parana_brl.parquet')
df_trigo_rs_brl = pd.read_parquet('../data/trigo/trigo_rs_brl.parquet')


dfs_brl = [df_acucar_santos_brl, df_acucar_sp_brl, df_algodao_brl, df_arroz_brl, df_cafe_arabica_brl, df_cafe_robusta_brl, df_milho_brl,
           df_soja_parana_brl, df_soja_paranagua_brl, df_trigo_parana_brl, df_trigo_rs_brl]

df_brl_final = merge_parquet_dfs(dfs_brl)

export_to_parquet(df_brl_final, "../data/all_comm/all_commodities_brl.parquet")