import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from typing import List, Dict, Optional, Union, Callable
import numpy as np
from pathlib import Path

class PlotRegistry:
    """Registry para tipos de plots disponíveis"""
    
    PLOTS = {}
    
    @classmethod
    def register_plot(cls, name: str, plot_func: Callable):
        """Registra um novo tipo de plot"""
        cls.PLOTS[name] = plot_func
    
    @classmethod
    def get_plot(cls, name: str) -> Callable:
        """Retorna função do plot"""
        if name not in cls.PLOTS:
            raise ValueError(f"Plot '{name}' não encontrado. Disponíveis: {list(cls.PLOTS.keys())}")
        return cls.PLOTS[name]
    
    @classmethod
    def list_plots(cls) -> List[str]:
        """Lista plots disponíveis"""
        return list(cls.PLOTS.keys())

class ForecastVisualizer:
    """Visualizador para previsões de séries temporais"""
    
    def __init__(
        self, 
        plot_types: Optional[List[str]] = None,
        save_plots: bool = False,
        output_dir: Optional[str] = None,
        figsize: tuple = (12, 6),
        style: str = 'seaborn-v0_8'
    ):
        """
        Inicializa o visualizador
        
        Args:
            plot_types: Lista de tipos de plots. Se None, usa todos disponíveis
            save_plots: Se True, salva plots em arquivos
            output_dir: Diretório para salvar plots
            figsize: Tamanho padrão das figuras
            style: Estilo do matplotlib
        """
        self.plot_types = plot_types or self._get_default_plots()
        self.save_plots = save_plots
        self.output_dir = Path(output_dir) if output_dir else Path('plots')
        self.figsize = figsize
        
        # Configurar estilo
        plt.style.use(style)
        
        # Criar diretório se necessário
        if self.save_plots:
            self.output_dir.mkdir(exist_ok=True)
        
        # Registrar plots padrão
        self._register_default_plots()
    
    def _get_default_plots(self) -> List[str]:
        """Retorna lista de plots padrão"""
        return [
            'metrics_comparison',
            'forecasts_grid', 
            'validation_forecasts',
            'test_forecasts',
            'residuals_analysis'
        ]
    
    def _register_default_plots(self):
        """Registra plots padrão"""
        PlotRegistry.register_plot('metrics_comparison', self._plot_metrics_comparison)
        PlotRegistry.register_plot('forecasts_grid', self._plot_forecasts_grid)
        PlotRegistry.register_plot('validation_forecasts', self._plot_validation_forecasts)
        PlotRegistry.register_plot('test_forecasts', self._plot_test_forecasts)
        PlotRegistry.register_plot('residuals_analysis', self._plot_residuals_analysis)
    
    def _plot_metrics_comparison(
    self, 
    metrics_df: pd.DataFrame, 
    metrics: Optional[List[str]] = None,
    **kwargs
    ) -> Figure:
        """Plota comparação de métricas entre modelos"""
        metrics = metrics or ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']
        available_metrics = [m for m in metrics if m in metrics_df.columns]
        
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Converter para array 1D
        if n_rows * n_cols == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()
        
        for idx, metric in enumerate(available_metrics):
            ax = axes_flat[idx]
            
            sns.barplot(
                data=metrics_df,
                x='unique_id' if 'unique_id' in metrics_df.columns else 'Commodity',
                y=metric,
                hue='Model',
                ax=ax
            )
            ax.set_title(f'{metric} por Commodity')
            ax.tick_params(axis='x', rotation=45)
        
        # Remove axes vazios
        for idx in range(n_metrics, len(axes_flat)):
            fig.delaxes(axes_flat[idx])
        
        plt.tight_layout()
        return fig
    
    def _plot_forecasts_grid(
    self,
    actual: pd.DataFrame,
    forecasts: pd.DataFrame,
    models: List[str],
    commodities: List[str],
    n_cols: int = 2,
    **kwargs
    ) -> Figure:
        """Plota grid de previsões para múltiplas commodities"""
        n_rows = (len(commodities) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        
        # Converter para array 1D
        if n_rows * n_cols == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()
        
        for idx, commodity in enumerate(commodities):
            ax = axes_flat[idx]
            
            # Plot valores reais
            actual_data = actual[actual['unique_id'] == commodity]
            ax.plot(
                actual_data['ds'],
                actual_data['y'],
                label='Valor Real',
                color='black',
                linewidth=2
            )
            
            # Plot previsões
            forecast_data = forecasts[forecasts['unique_id'] == commodity]
            for model in models:
                if model in forecast_data.columns:
                    ax.plot(
                        forecast_data['ds'],
                        forecast_data[model],
                        label=model,
                        alpha=0.8
                    )
            
            ax.set_title(commodity)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Remove axes vazios
        for idx in range(len(commodities), len(axes_flat)):
            fig.delaxes(axes_flat[idx])
        
        plt.tight_layout()
        
        # Pegar handles e labels do último axes que foi usado
        if commodities:
            handles, labels = axes_flat[len(commodities)-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(models)+1)
        
        return fig
    
    def _plot_validation_forecasts(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        forecasts_val: pd.DataFrame,
        commodity: str,
        models: Optional[List[str]] = None,
        **kwargs
    ) -> Figure:
        """Plota previsões de validação vs valores reais"""
        models = models or ['Naive', 'AutoARIMA']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Dados de treino
        train_commodity = train_data[train_data['unique_id'] == commodity]
        ax.plot(
            train_commodity['ds'],
            train_commodity['y'],
            label=f'Real {commodity} (Treino)',
            linestyle='--',
            color='gray',
            alpha=0.7
        )
        
        # Dados de validação
        val_commodity = val_data[val_data['unique_id'] == commodity]
        ax.plot(
            val_commodity['ds'],
            val_commodity['y'],
            label=f'Real {commodity} (Validação)',
            linestyle='-',
            color='black',
            linewidth=2
        )
        
        # Previsões
        forecast_commodity = forecasts_val[forecasts_val['unique_id'] == commodity]
        for model in models:
            if model in forecast_commodity.columns:
                ax.plot(
                    forecast_commodity['ds'],
                    forecast_commodity[model],
                    label=f'Previsão {model}',
                    alpha=0.8
                )
        
        ax.legend()
        ax.set_title(f'Previsões na Validação vs Valores Reais - {commodity}')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _plot_test_forecasts(
        self,
        full_train: pd.DataFrame,
        test_data: pd.DataFrame,
        forecasts_test: pd.DataFrame,
        commodity: str,
        models: Optional[List[str]] = None,
        **kwargs
    ) -> Figure:
        """Plota previsões de teste vs valores reais"""
        models = models or ['Naive', 'AutoARIMA']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Dados de treino + validação
        train_commodity = full_train[full_train['unique_id'] == commodity]
        ax.plot(
            train_commodity['ds'],
            train_commodity['y'],
            label=f'Real {commodity} (Treino + Validação)',
            linestyle='--',
            color='gray',
            alpha=0.7
        )
        
        # Dados de teste
        test_commodity = test_data[test_data['unique_id'] == commodity]
        ax.plot(
            test_commodity['ds'],
            test_commodity['y'],
            label=f'Real {commodity} (Teste)',
            linestyle='-',
            color='black',
            linewidth=2
        )
        
        # Previsões
        forecast_commodity = forecasts_test[forecasts_test['unique_id'] == commodity]
        for model in models:
            if model in forecast_commodity.columns:
                ax.plot(
                    forecast_commodity['ds'],
                    forecast_commodity[model],
                    label=f'Previsão {model}',
                    alpha=0.8
                )
        
        ax.legend()
        ax.set_title(f'Previsões no Teste vs Valores Reais - {commodity}')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _plot_residuals_analysis(
        self,
        actual: pd.DataFrame,
        forecasts: pd.DataFrame,
        model: str,
        commodity: str,
        **kwargs
    ) -> Figure:
        """Plota análise de resíduos"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Merge dados
        merged = forecasts.merge(actual, on=['ds', 'unique_id'])
        commodity_data = merged[merged['unique_id'] == commodity]
        
        residuals = commodity_data['y'] - commodity_data[model]
        
        # Plot 1: Resíduos vs Tempo
        axes[0, 0].scatter(commodity_data['ds'], residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Resíduos vs Tempo')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Resíduos vs Previsões
        axes[0, 1].scatter(commodity_data[model], residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_title('Resíduos vs Previsões')
        
        # Plot 3: Histograma dos resíduos
        axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribuição dos Resíduos')
        
        # Plot 4: Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        return fig
    
    def generate_plots(
        self,
        plot_type: str,
        save_name: Optional[str] = None,
        **kwargs
    ) -> Optional[Figure]:
        """
        Gera um plot específico
        
        Args:
            plot_type: Tipo do plot a ser gerado
            save_name: Nome do arquivo para salvar (se save_plots=True)
            **kwargs: Argumentos específicos para o plot
            
        Returns:
            Figura do matplotlib se não salvar, None se salvar
        """
        if plot_type not in self.plot_types:
            raise ValueError(f"Plot '{plot_type}' não está na lista de plots configurados")
        
        plot_func = PlotRegistry.get_plot(plot_type)
        fig = plot_func(**kwargs)
        
        if self.save_plots:
            filename = save_name or f"{plot_type}.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot salvo em: {filepath}")
            return None
        else:
            plt.show()
            return fig
    
    def generate_all_plots(
        self,
        data_dict: Dict,
        save_prefix: str = "",
        **kwargs
    ) -> None:
        """
        Gera todos os plots configurados
        
        Args:
            data_dict: Dicionário com dados necessários para os plots
            save_prefix: Prefixo para nomes dos arquivos
            **kwargs: Argumentos adicionais
        """
        for plot_type in self.plot_types:
            try:
                save_name = f"{save_prefix}{plot_type}.png" if save_prefix else None
                self.generate_plots(plot_type, save_name, **data_dict, **kwargs)
            except Exception as e:
                print(f"Erro ao gerar plot {plot_type}: {e}")
    
    def add_plot_type(self, name: str, plot_func: Callable) -> None:
        """Adiciona um novo tipo de plot"""
        PlotRegistry.register_plot(name, plot_func)
        if name not in self.plot_types:
            self.plot_types.append(name)