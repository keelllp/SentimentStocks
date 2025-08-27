"""
Model evaluation and visualization utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Model evaluation and visualization class"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various evaluation metrics"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'smape': 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        }
        
        return metrics
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  title: str = "Predictions vs Actual", 
                                  save_path: str = None) -> None:
        """Plot predictions vs actual values"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6, color='blue')
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{title} - Scatter Plot')
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{title} - Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_time_series_predictions(self, dates: pd.DatetimeIndex, y_true: np.ndarray, 
                                   y_pred: np.ndarray, title: str = "Time Series Predictions",
                                   save_path: str = None) -> None:
        """Plot time series predictions"""
        plt.figure(figsize=(15, 8))
        
        plt.plot(dates, y_true, label='Actual', linewidth=2, color='blue')
        plt.plot(dates, y_pred, label='Predicted', linewidth=2, color='red', linestyle='--')
        
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 20, title: str = "Feature Importance",
                              save_path: str = None) -> None:
        """Plot feature importance"""
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict], 
                            save_path: str = None) -> None:
        """Plot comparison of different models"""
        models = list(results.keys())
        metrics = ['rmse', 'mae', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def create_interactive_plots(self, dates: pd.DatetimeIndex, y_true: np.ndarray, 
                               y_pred_dict: Dict[str, np.ndarray], 
                               title: str = "Interactive Stock Price Predictions") -> go.Figure:
        """Create interactive Plotly plots"""
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions for each model
        colors = ['red', 'green', 'orange', 'purple']
        for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_pred,
                mode='lines',
                name=f'{model_name} Predicted',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Stock Price',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                              save_path: str = None) -> None:
        """Plot correlation matrix of features"""
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, train_sizes: np.ndarray, train_scores: np.ndarray, 
                           val_scores: np.ndarray, title: str = "Learning Curves",
                           save_path: str = None) -> None:
        """Plot learning curves"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', 
                color='blue', label='Training score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', 
                color='red', label='Cross-validation score')
        
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results: Dict[str, Dict], 
                                 save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("STOCK PRICE PREDICTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 30)
        
        for model_name, metrics in results.items():
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  RMSE: {metrics['rmse']:.4f}")
            report.append(f"  MAE:  {metrics['mae']:.4f}")
            report.append(f"  R²:   {metrics['r2']:.4f}")
        
        # Best model identification
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        report.append(f"\nBest Model (by RMSE): {best_model.upper()}")
        report.append(f"Best RMSE: {results[best_model]['rmse']:.4f}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS")
        report.append("-" * 20)
        
        for model_name, metrics in results.items():
            if metrics['r2'] < 0.5:
                report.append(f"  {model_name.upper()}: Consider feature engineering or hyperparameter tuning")
            elif metrics['r2'] < 0.7:
                report.append(f"  {model_name.upper()}: Moderate performance, room for improvement")
            else:
                report.append(f"  {model_name.upper()}: Good performance")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text
    
    def save_all_plots(self, results: Dict[str, Dict], dates: pd.DatetimeIndex,
                       y_true: np.ndarray, y_pred_dict: Dict[str, np.ndarray],
                       feature_importance: pd.DataFrame = None) -> None:
        """Save all evaluation plots"""
        # Model comparison
        self.plot_model_comparison(results, 
                                 os.path.join(self.results_dir, 'model_comparison.png'))
        
        # Time series predictions
        self.plot_time_series_predictions(dates, y_true, y_pred_dict[list(y_pred_dict.keys())[0]], 
                                        save_path=os.path.join(self.results_dir, 'time_series_predictions.png'))
        
        # Feature importance (if available)
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance, 
                                       save_path=os.path.join(self.results_dir, 'feature_importance.png'))
        
        # Generate report
        self.generate_evaluation_report(results, 
                                     os.path.join(self.results_dir, 'evaluation_report.txt'))
        
        print(f"All plots and report saved to: {self.results_dir}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    y_true = np.random.randn(n_samples).cumsum() + 100
    y_pred_xgb = y_true + np.random.randn(n_samples) * 2
    y_pred_lgb = y_true + np.random.randn(n_samples) * 1.5
    
    # Sample results
    results = {
        'XGBoost': {'rmse': 2.1, 'mae': 1.8, 'r2': 0.85},
        'LightGBM': {'rmse': 1.9, 'mae': 1.6, 'r2': 0.87}
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Generate plots
    evaluator.plot_model_comparison(results)
    evaluator.plot_time_series_predictions(dates, y_true, y_pred_xgb)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    print(report)
