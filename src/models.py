"""
Machine learning models for stock price prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class StockPricePredictor:
    """Main class for stock price prediction using XGBoost and LightGBM"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data (time series split for financial data)
        split_idx = int(len(X_array) * (1 - test_size))
        X_train, X_test = X_array[:split_idx], X_array[split_idx:]
        y_train, y_test = y_array[:split_idx], y_array[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None,
                      **params) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        # Default parameters
        default_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
        
        # Update with custom parameters
        default_params.update(params)
        
        # Initialize model
        self.xgb_model = xgb.XGBRegressor(**default_params)
        
        # Train model
        if X_val is not None and y_val is not None:
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100
            )
        else:
            self.xgb_model.fit(X_train, y_train, verbose=100)
        
        return self.xgb_model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       **params) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        # Default parameters
        default_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'metric': 'rmse',
            'verbose': -1
        }
        
        # Update with custom parameters
        default_params.update(params)
        
        # Initialize model
        self.lgb_model = lgb.LGBMRegressor(**default_params)
        
        # Train model
        if X_val is not None and y_val is not None:
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
        else:
            self.lgb_model.fit(X_train, y_train)
        
        return self.lgb_model
    
    def train_models(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2,
                    xgb_params: Dict = None,
                    lgb_params: Dict = None) -> Dict[str, Any]:
        """Train both XGBoost and LightGBM models"""
        print("Preparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, test_size)
        
        # Split training data for validation
        val_split_idx = int(len(X_train) * 0.8)
        X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        print("Training XGBoost model...")
        self.train_xgboost(X_train_final, y_train_final, X_val, y_val, **(xgb_params or {}))
        
        print("Training LightGBM model...")
        self.train_lightgbm(X_train_final, y_train_final, X_val, y_val, **(lgb_params or {}))
        
        # Evaluate models
        print("Evaluating models...")
        results = self.evaluate_models(X_test, y_test)
        
        self.is_trained = True
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        results = {}
        
        # XGBoost evaluation
        if self.xgb_model is not None:
            y_pred_xgb = self.xgb_model.predict(X_test)
            results['xgboost'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
                'mae': mean_absolute_error(y_test, y_pred_xgb),
                'r2': r2_score(y_test, y_pred_xgb),
                'predictions': y_pred_xgb
            }
        
        # LightGBM evaluation
        if self.lgb_model is not None:
            y_pred_lgb = self.lgb_model.predict(X_test)
            results['lightgbm'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lgb)),
                'mae': mean_absolute_error(y_test, y_pred_lgb),
                'r2': r2_score(y_test, y_pred_lgb),
                'predictions': y_pred_lgb
            }
        
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        X_array = self.scaler.fit_transform(X.values)
        y_array = y.values
        
        # Time series split for financial data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_results = {'xgboost': [], 'lightgbm': []}
        
        for train_idx, val_idx in tscv.split(X_array):
            X_train_cv, X_val_cv = X_array[train_idx], X_array[val_idx]
            y_train_cv, y_val_cv = y_array[train_idx], y_array[val_idx]
            
            # XGBoost CV
            xgb_cv = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_cv.fit(X_train_cv, y_train_cv)
            y_pred_xgb = xgb_cv.predict(X_val_cv)
            cv_results['xgboost'].append(r2_score(y_val_cv, y_pred_xgb))
            
            # LightGBM CV
            lgb_cv = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_cv.fit(X_train_cv, y_train_cv)
            y_pred_lgb = lgb_cv.predict(X_val_cv)
            cv_results['lightgbm'].append(r2_score(y_val_cv, y_pred_lgb))
        
        return cv_results
    
    def get_feature_importance(self, model_type: str = 'both') -> Dict[str, pd.DataFrame]:
        """Get feature importance from trained models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before getting feature importance")
        
        importance_dict = {}
        
        if model_type in ['both', 'xgboost'] and self.xgb_model is not None:
            xgb_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_dict['xgboost'] = xgb_importance
        
        if model_type in ['both', 'lightgbm'] and self.lgb_model is not None:
            lgb_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.lgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_dict['lightgbm'] = lgb_importance
        
        return importance_dict
    
    def predict(self, X: pd.DataFrame, model_type: str = 'ensemble') -> np.ndarray:
        """Make predictions using trained models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X.values)
        
        if model_type == 'xgboost' and self.xgb_model is not None:
            return self.xgb_model.predict(X_scaled)
        elif model_type == 'lightgbm' and self.lgb_model is not None:
            return self.lgb_model.predict(X_scaled)
        elif model_type == 'ensemble' and self.xgb_model is not None and self.lgb_model is not None:
            # Ensemble prediction (simple average)
            xgb_pred = self.xgb_model.predict(X_scaled)
            lgb_pred = self.lgb_model.predict(X_scaled)
            return (xgb_pred + lgb_pred) / 2
        else:
            raise ValueError(f"Model type '{model_type}' not available")
    
    def save_models(self, filename_prefix: str = "stock_predictor") -> None:
        """Save trained models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        # Save XGBoost model
        if self.xgb_model is not None:
            xgb_path = os.path.join(self.models_dir, f"{filename_prefix}_xgboost.pkl")
            joblib.dump(self.xgb_model, xgb_path)
            print(f"XGBoost model saved to: {xgb_path}")
        
        # Save LightGBM model
        if self.lgb_model is not None:
            lgb_path = os.path.join(self.models_dir, f"{filename_prefix}_lightgbm.pkl")
            joblib.dump(self.lgb_model, lgb_path)
            print(f"LightGBM model saved to: {lgb_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f"{filename_prefix}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save feature names
        features_path = os.path.join(self.models_dir, f"{filename_prefix}_features.pkl")
        joblib.dump(self.feature_names, features_path)
        print(f"Feature names saved to: {features_path}")
    
    def load_models(self, filename_prefix: str = "stock_predictor") -> None:
        """Load trained models"""
        # Load XGBoost model
        xgb_path = os.path.join(self.models_dir, f"{filename_prefix}_xgboost.pkl")
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            print(f"XGBoost model loaded from: {xgb_path}")
        
        # Load LightGBM model
        lgb_path = os.path.join(self.models_dir, f"{filename_prefix}_lightgbm.pkl")
        if os.path.exists(lgb_path):
            self.lgb_model = joblib.load(lgb_path)
            print(f"LightGBM model loaded from: {lgb_path}")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, f"{filename_prefix}_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from: {scaler_path}")
        
        # Load feature names
        features_path = os.path.join(self.models_dir, f"{filename_prefix}_features.pkl")
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
            print(f"Feature names loaded from: {features_path}")
        
        self.is_trained = True

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X_sample = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_sample = np.random.randn(n_samples) + 100
    
    # Initialize predictor
    predictor = StockPricePredictor()
    
    # Train models
    print("Training models...")
    results = predictor.train_models(X_sample, y_sample)
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    print(f"\nTop 5 features by importance:")
    for model_name, importance_df in importance.items():
        print(f"\n{model_name.upper()}:")
        print(importance_df.head())
    
    # Save models
    predictor.save_models("sample_predictor")
