import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union


class BasePreprocessor(ABC):
    """Classe base para todos os preprocessadores."""
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação no DataFrame."""
        pass


class CurrencyExtractor(BasePreprocessor):
    """Extrai séries temporais por moeda."""
    
    def __init__(self, currency: str = "BRL", commodity_type: str = "standard"):
        self.currency = currency
        self.commodity_type = commodity_type
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrai a série temporal da moeda especificada.
        
        Args:
            df: DataFrame contendo colunas de moedas BRL e USD
            
        Returns:
            DataFrame com colunas 'Data' e valor da moeda selecionada
        """
        if self.commodity_type == "algodao":
            return self._extract_algodao_series(df)
        else:
            return self._extract_standard_series(df)
    
    def _extract_standard_series(self, df: pd.DataFrame) -> Any:
        """Extrai série para commodities padrão."""
        if self.currency == "BRL":
            return df[['Data', 'À vista R$']].copy()
        elif self.currency == "USD":
            return df[['Data', 'À vista US$']].copy()
        else:
            raise ValueError(f"Moeda inválida: {self.currency}")
    
    def _extract_algodao_series(self, df: pd.DataFrame) -> Any:
        """Extrai série específica para algodão."""
        if self.currency == "BRL":
            return df[['Data', 'Prazo de 8 dias R$']].copy()
        elif self.currency == "USD":
            return df[['Data', 'Prazo de 8 dias US$']].copy()
        else:
            raise ValueError(f"Moeda inválida: {self.currency}")


class CurrencySplitter(BasePreprocessor):
    """Separa DataFrame em séries BRL e USD."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATENÇÃO: Esta classe retorna apenas o primeiro DataFrame da tuple.
        Use split_currencies() para obter ambos DataFrames.
        
        Args:
            df: DataFrame contendo as colunas 'Data', 'À vista R$', e 'À vista US$'
            
        Returns:
            DataFrame com dados em reais (primeiro da tuple)
        """
        df_brl, _ = self.split_currencies(df)
        return df_brl
    
    def split_currencies(self, df: pd.DataFrame) -> Any:
        """
        Separa um DataFrame com colunas de moedas BRL e USD em dois DataFrames distintos.
        
        Args:
            df: DataFrame contendo as colunas 'Data', 'À vista R$', e 'À vista US$'
            
        Returns:
            Tuple com dois DataFrames: um com dados em reais e outro em dólares
        """
        df_brl = df[['Data', 'À vista R$']].copy()
        df_usd = df[['Data', 'À vista US$']].copy()
        return df_brl, df_usd


class ColumnRenamer(BasePreprocessor):
    """Renomeia colunas e adiciona identificador único."""
    
    def __init__(self, cols_dict: Dict[str, str], unique_id: str):
        self.cols_dict = cols_dict
        self.unique_id = unique_id
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renomeia colunas e adiciona um identificador único ao DataFrame.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame com colunas renomeadas e nova coluna 'unique_id'
        """
        df = df.rename(columns=self.cols_dict, inplace=False)
        df['unique_id'] = self.unique_id
        return df


class DateTimeConverter(BasePreprocessor):
    """Converte colunas para datetime."""
    
    def __init__(self, column: str, date_format: str = '%d/%m/%Y'):
        self.column = column
        self.date_format = date_format
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte a coluna alvo de strings para objetos datetime.
        
        Args:
            df: DataFrame contendo a coluna alvo com datas em formato string
            
        Returns:
            DataFrame com a coluna convertida para datetime
        """
        df = df.copy()
        df[self.column] = pd.to_datetime(df[self.column], format=self.date_format, errors='coerce')
        return df


class FloatConverter(BasePreprocessor):
    """Converte colunas para float."""
    
    def __init__(self, column: str):
        self.column = column
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte a coluna alvo de string para float, tratando vírgulas como separadores decimais.
        
        Args:
            df: DataFrame contendo a coluna alvo com valores numéricos como strings
            
        Returns:
            DataFrame com a coluna alvo convertida para tipo float
        """
        df = df.copy()
        df[self.column] = df[self.column].astype(str).str.replace(',', '.', regex=False)
        df[self.column] = df[self.column].astype(float)
        return df


class MonthlyAggregator(BasePreprocessor):
    """Agrega dados por média mensal."""
    
    def transform(self, df: pd.DataFrame) -> Any:
        """
        Agrega a série temporal calculando a média mensal por identificador único.
        
        Args:
            df: DataFrame contendo colunas 'ds' (datetime), 'y' (valor), e 'unique_id'
            
        Returns:
            DataFrame com a média mensal por 'unique_id' no formato adequado para modelagem temporal
        """
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['year_month'] = df['ds'].dt.to_period('M')

        df_monthly = df.groupby(['year_month', 'unique_id']).agg({'y': 'mean'}).reset_index()
        df_monthly['ds'] = pd.to_datetime(df_monthly['year_month'].astype(str) + '-01')
        df_monthly = df_monthly[['ds', 'y', 'unique_id']]
        
        return df_monthly


class DateFilter(BasePreprocessor):
    """Remove linhas com datas posteriores a uma data limite."""
    
    def __init__(self, limit_date: str, column_date: str, date_format: str = '%d/%m/%Y'):
        self.limit_date = limit_date
        self.column_date = column_date
        self.date_format = date_format
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove as linhas do DataFrame cuja data em uma coluna especificada excede uma data limite.
        
        Args:
            df: DataFrame contendo uma coluna de datas
            
        Returns:
            DataFrame filtrado com apenas as linhas anteriores à data limite
        """
        limit_date = pd.to_datetime(self.limit_date, format=self.date_format)
        df = df[df[self.column_date] < limit_date] # type: ignore
        return df


class ParquetExporter:
    """Salva DataFrame em formato Parquet."""
    
    @staticmethod
    def export(df: pd.DataFrame, path_parquet: str) -> None:
        """
        Salva um DataFrame em formato Parquet no caminho especificado.
        
        Args:
            df: DataFrame a ser salvo
            path_parquet: Caminho completo para salvar o arquivo .parquet
        """
        dir_final = os.path.dirname(path_parquet)
        if not os.path.exists(dir_final):
            os.makedirs(dir_final)

        df.to_parquet(path_parquet, engine="pyarrow")
        print(f'Arquivo salvo em: {path_parquet}')


class PreprocessingPipeline:
    """Pipeline de preprocessamento que aplica múltiplas transformações."""
    
    def __init__(self, preprocessors: list):
        self.preprocessors = preprocessors
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todos os preprocessadores na ordem especificada.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame após todas as transformações
        """
        result = df.copy()
        for preprocessor in self.preprocessors:
            result = preprocessor.transform(result)
        return result
    
    def add_step(self, preprocessor: BasePreprocessor) -> None:
        """Adiciona um novo preprocessador ao pipeline."""
        self.preprocessors.append(preprocessor)
    
    def remove_step(self, index: int) -> None:
        """Remove um preprocessador do pipeline."""
        if 0 <= index < len(self.preprocessors):
            self.preprocessors.pop(index)


# Funções de compatibilidade com código legado
def split_currency_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Função de compatibilidade para separar dados de moedas."""
    splitter = CurrencySplitter()
    return splitter.split_currencies(df)


def extract_currency_series(df: pd.DataFrame, currency: str = "BRL") -> pd.DataFrame:
    """Função de compatibilidade para extrair série de moeda."""
    extractor = CurrencyExtractor(currency=currency)
    return extractor.transform(df)


def extract_currency_series_algodao(df: pd.DataFrame, currency: str = "BRL") -> pd.DataFrame:
    """Função de compatibilidade para extrair série de algodão."""
    extractor = CurrencyExtractor(currency=currency, commodity_type="algodao")
    return extractor.transform(df)


def rename_columns_and_set_id(df: pd.DataFrame, cols_dict: Dict[str, str], unique_id: str) -> pd.DataFrame:
    """Função de compatibilidade para renomear colunas."""
    renamer = ColumnRenamer(cols_dict, unique_id)
    return renamer.transform(df)


def convert_column_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Função de compatibilidade para converter datas."""
    converter = DateTimeConverter(column)
    return converter.transform(df)


def convert_column_to_float(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Função de compatibilidade para converter floats."""
    converter = FloatConverter(column)
    return converter.transform(df)


def aggregate_monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Função de compatibilidade para agregação mensal."""
    aggregator = MonthlyAggregator()
    return aggregator.transform(df)


def drop_over_limit_date(df: pd.DataFrame, limit_date: str, column_date: str) -> pd.DataFrame:
    """Função de compatibilidade para filtrar datas."""
    filter_obj = DateFilter(limit_date, column_date)
    return filter_obj.transform(df)


def export_to_parquet(df: pd.DataFrame, path_parquet: str) -> None:
    """Função de compatibilidade para exportar parquet."""
    ParquetExporter.export(df, path_parquet) 