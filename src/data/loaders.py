import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from .preprocessors import (
    CurrencyExtractor, 
    ColumnRenamer, 
    DateTimeConverter, 
    FloatConverter, 
    MonthlyAggregator,
    DateFilter,
    MonthlyFirstAggregator,
    MonthlyLastAggregator,
    PreprocessingPipeline
)
from src.utils.find_root import get_project_root



class CommodityLoader:
    """Carregador unificado para commodities."""
    
    
    COMMODITY_CONFIGS = {
        'acucar_santos': {
            'data_path': 'data/raw/acucar/Indicador Açúcar Cristal - Santos (FOB).csv',
            'unique_id': 'ACUCAR_SANTOS',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'acucar_sp': {
            'data_path': 'data/raw/acucar/INDICADOR DO AÇÚCAR CRISTAL BRANCO CEPEA-ESALQ - SÃO PAULO.csv',
            'unique_id': 'ACUCAR_SP',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'algodao': {
            'data_path': 'data/raw/algodao/Indicador do Algodão em Pluma CEPEA-ESALQ - Prazo de 8 dias.csv',
            'unique_id': 'ALGODAO',
            'currency_columns': {'BRL': 'Prazo de 8 dias R$', 'USD': 'Prazo de 8 dias US$'},
            'date_column': 'Data',
            'commodity_type': 'algodao',
            'separator': ','
        },
        'arroz': {
            'data_path': 'data/raw/arroz/INDICADOR DO ARROZ EM CASCA CEPEA-IRGA-RS.csv',
            'unique_id': 'ARROZ',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'cafe_arabica': {
            'data_path': 'data/raw/cafe/INDICADOR DO CAFÉ ARÁBICA CEPEA-ESALQ.csv',
            'unique_id': 'CAFE_ARABICA',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'cafe_robusta': {
            'data_path': 'data/raw/cafe/INDICADOR DO CAFÉ ROBUSTA CEPEA-ESALQ.csv',
            'unique_id': 'CAFE_ROBUSTA',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'milho': {
            'data_path': 'data/raw/milho/INDICADOR DO MILHO ESALQ-BM&FBOVESPA.csv',
            'unique_id': 'MILHO',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'soja_parana': {
            'data_path': 'data/raw/soja/INDICADOR DA SOJA CEPEA-ESALQ - PARANÁ.csv',
            'unique_id': 'SOJA_PARANA',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'soja_paranagua': {
            'data_path': 'data/raw/soja/INDICADOR DA SOJA CEPEA-ESALQ - PARANAGUÁ.csv',
            'unique_id': 'SOJA_PARANAGUA',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'trigo_parana': {
            'data_path': 'data/raw/trigo/PREÇO MÉDIO DO TRIGO CEPEA-ESALQ - PARANÁ.csv',
            'unique_id': 'TRIGO_PARANA',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        },
        'trigo_rs': {
            'data_path': 'data/raw/trigo/PREÇO MÉDIO DO TRIGO CEPEA-ESALQ - RIO GRANDE DO SUL.csv',
            'unique_id': 'TRIGO_RS',
            'currency_columns': {'BRL': 'À vista R$', 'USD': 'À vista US$'},
            'date_column': 'Data',
            'commodity_type': 'standard',
            'separator': ','
        }
    }
    
    @classmethod
    def _get_absolute_path(cls, relative_path: str) -> str:
        """
        Converte caminho relativo para absoluto baseado no root do projeto.
        
        Args:
            relative_path: Caminho relativo a partir do root do projeto
            
        Returns:
            Caminho absoluto para o arquivo
        """
        project_root = get_project_root()
        return os.path.join(project_root, relative_path)
    
    @classmethod
    def load_commodity(cls, commodity_name: str, currency: str = 'BRL', 
                      preprocessing: bool = True, monthly_aggregation: Optional[str] = "mean",
                      limit_date: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega uma commodity específica com preprocessamento automático.
        
        Args:
            commodity_name: Nome da commodity (ex: 'acucar_santos', 'milho', etc.)
            currency: Moeda desejada ('BRL' ou 'USD')
            preprocessing: Se deve aplicar preprocessamento básico
            monthly_aggregation: Tipo de agregação mensal:
                - "mean"  → média (padrão)
                - "first" → primeiro dia do mês
                - "last"  → último dia do mês
                - None    → sem agregação
            limit_date: Data limite para filtrar dados (formato '%d/%m/%Y')
            
        Returns:
            DataFrame processado com colunas ['ds', 'y', 'unique_id']
        """
        if commodity_name not in cls.COMMODITY_CONFIGS:
            raise ValueError(f"Commodity '{commodity_name}' não encontrada. "
                           f"Disponíveis: {list(cls.COMMODITY_CONFIGS.keys())}")
        
        config = cls.COMMODITY_CONFIGS[commodity_name]
        absolute_path = cls._get_absolute_path(config['data_path'])
        
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {absolute_path}")
        
        df = pd.read_csv(absolute_path, sep=config.get('separator', ','))
        
        if not preprocessing:
            return df
        
        # pipeline
        preprocessors = []
        
        currency_extractor = CurrencyExtractor(
            currency=currency, 
            commodity_type=config['commodity_type']
        )
        df = currency_extractor.transform(df)
        
        cols_dict = {
            config['date_column']: 'ds',
            config['currency_columns'][currency]: 'y'
        }
        column_renamer = ColumnRenamer(cols_dict, config['unique_id'])
        preprocessors.append(column_renamer)
        
        datetime_converter = DateTimeConverter('ds')
        preprocessors.append(datetime_converter)
        
        float_converter = FloatConverter('y')
        preprocessors.append(float_converter)
        
        if limit_date:
            date_filter = DateFilter(limit_date, 'ds')
            preprocessors.append(date_filter)
        
        if monthly_aggregation:
            if monthly_aggregation == "mean":
                aggregator = MonthlyAggregator()
            elif monthly_aggregation == "first":
                aggregator = MonthlyFirstAggregator()
            elif monthly_aggregation == "last":
                aggregator = MonthlyLastAggregator()
            else:
                raise ValueError("monthly_aggregation deve ser 'mean', 'first', 'last' ou None")
            preprocessors.append(aggregator)
        
        pipeline = PreprocessingPipeline(preprocessors)
        df = pipeline.fit_transform(df)
        
        return df
    
    @classmethod
    def load_all_commodities(cls, currency: str = 'BRL', 
                           preprocessing: bool = True, 
                           monthly_aggregation: Optional[str] = "mean",
                           limit_date: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega todas as commodities e concatena em um DataFrame único.
        
        Args:
            currency: Moeda desejada ('BRL' ou 'USD')
            preprocessing: Se deve aplicar preprocessamento básico
            monthly_aggregation: Se deve fazer agregação mensal
            limit_date: Data limite para filtrar dados
            
        Returns:
            DataFrame concatenado com todas as commodities
        """
        dfs = []
        
        for commodity_name in cls.COMMODITY_CONFIGS.keys():
            try:
                df = cls.load_commodity(
                    commodity_name=commodity_name,
                    currency=currency,
                    preprocessing=preprocessing,
                    monthly_aggregation=monthly_aggregation,
                    limit_date=limit_date
                )
                dfs.append(df)
                print(f"ദ്ദി・ᴗ・)✧ {commodity_name} carregado com sucesso")
            except Exception as e:
                print(f"(⁠╥⁠﹏⁠╥⁠) Erro ao carregar {commodity_name}: {e}")
                continue
        
        if not dfs:
            raise ValueError("Nenhuma commodity foi carregada com sucesso")
        
        result = pd.concat(dfs, axis=0, ignore_index=True)
        result = result.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        return result
    
    @classmethod
    def load_multiple_commodities(cls, commodity_names: List[str], 
                                currency: str = 'BRL',
                                preprocessing: bool = True,
                                monthly_aggregation: Optional[str] = "mean",
                                limit_date: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega múltiplas commodities específicas.
        
        Args:
            commodity_names: Lista de nomes de commodities
            currency: Moeda desejada ('BRL' ou 'USD')
            preprocessing: Se deve aplicar preprocessamento básico
            monthly_aggregation: Se deve fazer agregação mensal
            limit_date: Data limite para filtrar dados
            
        Returns:
            DataFrame concatenado com as commodities selecionadas
        """
        dfs = []
        
        for commodity_name in commodity_names:
            try:
                df = cls.load_commodity(
                    commodity_name=commodity_name,
                    currency=currency,
                    preprocessing=preprocessing,
                    monthly_aggregation=monthly_aggregation,
                    limit_date=limit_date
                )
                dfs.append(df)
                print(f"ദ്ദി・ᴗ・)✧ {commodity_name} carregado com sucesso")
            except Exception as e:
                print(f"(⁠╥⁠﹏⁠╥⁠) Erro ao carregar {commodity_name}: {e}")
                continue
        
        if not dfs:
            raise ValueError("Nenhuma commodity foi carregada com sucesso")
        
        result = pd.concat(dfs, axis=0, ignore_index=True)
        result = result.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        return result
    
    @classmethod
    def get_available_commodities(cls) -> List[str]:
        """Retorna lista de commodities disponíveis."""
        return list(cls.COMMODITY_CONFIGS.keys())
    
    @classmethod
    def get_commodity_info(cls, commodity_name: str) -> Dict:
        """Retorna informações sobre uma commodity específica."""
        if commodity_name not in cls.COMMODITY_CONFIGS:
            raise ValueError(f"Commodity '{commodity_name}' não encontrada")
        
        config = cls.COMMODITY_CONFIGS[commodity_name].copy()
        absolute_path = cls._get_absolute_path(config['data_path'])
        config['file_exists'] = os.path.exists(absolute_path)
        config['absolute_path'] = absolute_path
        
        return config


# Funções de conveniência para compatibilidade com código existente
def load_commodity_data(commodity_name: str, currency: str = 'BRL') -> pd.DataFrame:
    """
    Função de conveniência para carregar uma commodity.
    
    Args:
        commodity_name: Nome da commodity
        currency: Moeda ('BRL' ou 'USD')
        
    Returns:
        DataFrame processado
    """
    return CommodityLoader.load_commodity(commodity_name, currency)


def load_all_commodities_data(currency: str = 'BRL') -> pd.DataFrame:
    """
    Função de conveniência para carregar todas as commodities.
    
    Args:
        currency: Moeda ('BRL' ou 'USD')
        
    Returns:
        DataFrame concatenado
    """
    return CommodityLoader.load_all_commodities(currency) 