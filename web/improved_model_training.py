# ================================================================
# IMPROVED MODEL TRAINING FOR INDIAN STOCKS
# ================================================================
# File: improved_model_training.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ImprovedStockPredictor:
    def __init__(self):
        """Initialize the improved predictor with advanced models"""
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importance = {}
        
    def clean_and_prepare_data(self, df):
        """Clean and prepare data for training - handle string to float conversion"""
        df = df.copy()
        
        print(f"🔧 Data cleaning started...")
        print(f"Original data shape: {df.shape}")
        print(f"Original data types:\n{df.dtypes.value_counts()}")
        
        # Handle date columns
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Handle symbol/name columns (categorical)
        categorical_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's actually numeric data stored as string
                try:
                    pd.to_numeric(df[col], errors='raise')
                    # If successful, convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"✅ Converted {col} from string to numeric")
                except:
                    # If not numeric, treat as categorical
                    categorical_columns.append(col)
                    print(f"📝 Keeping {col} as categorical")
        
        # Convert categorical columns to numeric using one-hot encoding
        if categorical_columns:
            print(f"🔄 Converting {len(categorical_columns)} categorical columns to numeric...")
            df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            print(f"✅ One-hot encoding completed. New shape: {df_encoded.shape}")
        else:
            df_encoded = df
        
        # Remove any remaining non-numeric columns
        numeric_columns = df_encoded.select_dtypes(include=[np.number]).columns
        non_numeric_columns = df_encoded.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_columns) > 0:
            print(f"⚠️  Removing {len(non_numeric_columns)} non-numeric columns: {list(non_numeric_columns)}")
            df_encoded = df_encoded[numeric_columns]
        
        # Handle infinite values
        df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        print(f"🔧 Handling missing values...")
        print(f"Missing values before cleaning:\n{df_encoded.isnull().sum().sum()}")
        
        # For numeric columns, fill with median
        for col in df_encoded.columns:
            if df_encoded[col].dtype in ['float64', 'int64']:
                median_val = df_encoded[col].median()
                df_encoded[col] = df_encoded[col].fillna(median_val)
        
        print(f"Missing values after cleaning:\n{df_encoded.isnull().sum().sum()}")
        print(f"✅ Data cleaning completed. Final shape: {df_encoded.shape}")
        
        return df_encoded
        
    def create_advanced_features(self, df):
        """Create advanced technical and market features"""
        df = df.copy()
        
        print(f"🔧 Creating advanced features...")
        
        # Ensure we have required columns
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Change'] = df['Close'] - df['Close'].shift(1)
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_Ratio_{window}'] = df['Close'] / df[f'MA_{window}']
        
        # Volatility features
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['Volatility_Ratio'] = df['Volatility_5'] / df['Volatility_20']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            df['Price_Volume_Trend'] = df['Close'] * df['Volume']
        
        # Price momentum
        for period in [1, 3, 5, 10]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Support and resistance
        df['Support_Level'] = df['Close'].rolling(window=20).min()
        df['Resistance_Level'] = df['Close'].rolling(window=20).max()
        df['Support_Distance'] = (df['Close'] - df['Support_Level']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
        
        # Market regime features
        df['Trend_Strength'] = abs(df['MA_20'] - df['MA_100']) / df['MA_100']
        df['Market_Regime'] = np.where(df['MA_20'] > df['MA_100'], 1, -1)
        
        # Sector-specific features (if available)
        if 'Sector' in df.columns:
            df = self.add_sector_features(df)
        
        print(f"✅ Advanced features created. New shape: {df.shape}")
        return df
    
    def add_sector_features(self, df):
        """Add sector-specific features"""
        # Banking sector features
        if 'Sector' in df.columns and 'Banking' in df.columns:
            df['Capital_Adequacy'] = df.get('Capital', 0) / df.get('Risk_Assets', 1)
            df['Asset_Quality'] = df.get('NPA_Ratio', 0)
        
        # IT sector features
        if 'Sector' in df.columns and 'IT' in df.columns:
            df['Revenue_Growth'] = df.get('Revenue', 0).pct_change()
            df['Profit_Margin'] = df.get('Net_Income', 0) / df.get('Revenue', 1)
        
        return df
    
    def create_target_variables(self, df, target_horizons=[1, 3, 5]):
        """Create multiple target variables for different prediction horizons"""
        df = df.copy()
        
        print(f"🎯 Creating target variables...")
        
        for horizon in target_horizons:
            df[f'Target_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1
            df[f'Target_Direction_{horizon}d'] = np.where(df[f'Target_{horizon}d'] > 0, 1, 0)
        
        print(f"✅ Target variables created")
        return df
    
    def prepare_features(self, df, target_col='Target_1d'):
        """Prepare features for training"""
        print(f"📊 Preparing features for training...")
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        print(f"Data shape after removing NaN: {df_clean.shape}")
        
        # Separate features and target
        exclude_cols = ['Target_1d', 'Target_3d', 'Target_5d', 'Target_Direction_1d', 
                       'Target_Direction_3d', 'Target_Direction_5d', 'Date', 'Symbol']
        
        # Only exclude columns that actually exist
        existing_exclude_cols = [col for col in exclude_cols if col in df_clean.columns]
        feature_cols = [col for col in df_clean.columns if col not in existing_exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Final data type check
        print(f"🔍 Final data type check:")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Feature data types:\n{X.dtypes.value_counts()}")
        
        # Ensure all features are numeric
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                print(f"⚠️  Converting {col} to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Remove any remaining NaN values
        X = X.fillna(0)  # Fill with 0 as last resort
        y = y.fillna(0)
        
        print(f"✅ Feature preparation completed")
        return X, y, feature_cols
    
    def create_advanced_models(self):
        """Create advanced ensemble models"""
        # Base models with optimized parameters
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # Advanced ensemble with voting
        ensemble = VotingRegressor([
            ('rf', rf),
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('gb', gb_model)
        ], weights=[0.25, 0.3, 0.25, 0.2])
        
        self.models = {
            'random_forest': rf,
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'gradient_boosting': gb_model,
            'ensemble': ensemble
        }
        
        return self.models
    
    def optimize_hyperparameters(self, X, y, model_name='random_forest'):
        """Optimize hyperparameters using time-series cross-validation"""
        model = self.models[model_name]
        
        # Define parameter grids for different models
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10]
            }
        elif model_name == 'xgboost':
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.03, 0.05, 0.07],
                'subsample': [0.7, 0.8, 0.9]
            }
        else:
            return model  # Skip optimization for other models
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Grid search with time-series CV
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return self.models[model_name]
    
    def feature_selection(self, X, y, method='mutual_info'):
        """Perform feature selection"""
        if method == 'mutual_info':
            selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
        elif method == 'recursive':
            selector = RFE(estimator=RandomForestRegressor(n_estimators=100), n_features_to_select=50)
        else:
            return X, list(X.columns)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = list(X.columns[selector.get_support()])
        
        # Store feature importance
        if hasattr(selector, 'scores_'):
            feature_scores = pd.DataFrame({
                'feature': X.columns,
                'score': selector.scores_
            }).sort_values('score', ascending=False)
            self.feature_importance[method] = feature_scores
        
        return X_selected, selected_features
    
    def train_models(self, X, y, optimize=True):
        """Train all models with advanced techniques"""
        print("🚀 Training advanced models...")
        
        # Create models
        self.create_advanced_models()
        
        # Feature selection
        print("🔍 Performing feature selection...")
        X_selected, selected_features = self.feature_selection(X, y)
        print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
        
        # Store selected features
        self.selected_features = selected_features
        
        # Train each model
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            
            # Optimize hyperparameters if requested
            if optimize and name in ['random_forest', 'xgboost']:
                model = self.optimize_hyperparameters(X_selected, y, name)
            
            # Train model
            model.fit(X_selected, y)
            
            # Make predictions
            y_pred = model.predict(X_selected)
            
            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  R²: {r2:.4f}")
        
        return self.models
    
    def predict_ensemble(self, X):
        """Make ensemble prediction with confidence intervals"""
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name != 'ensemble':
                pred = model.predict(X)
                predictions[name] = pred
        
        # Calculate ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Calculate prediction confidence (standard deviation across models)
        pred_std = np.std(list(predictions.values()), axis=0)
        
        # Calculate confidence intervals
        confidence_95 = 1.96 * pred_std
        confidence_68 = 1.0 * pred_std
        
        return {
            'ensemble_prediction': ensemble_pred,
            'confidence_95': confidence_95,
            'confidence_68': confidence_68,
            'model_predictions': predictions,
            'prediction_std': pred_std
        }
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance with multiple metrics"""
        # Make predictions
        predictions = self.predict_ensemble(X_test)
        y_pred = predictions['ensemble_prediction']
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        # Direction accuracy
        pred_direction = np.diff(y_pred) > 0
        actual_direction = np.diff(y_test) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction)
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print(f"R² Score: {r2:.4f}")
        print(f"Direction Accuracy: {direction_accuracy:.2%}")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if 'mutual_info' not in self.feature_importance:
            print("No feature importance data available")
            return
        
        importance_df = self.feature_importance['mutual_info'].head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df)), importance_df['score'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def save_models(self, filepath_prefix="improved_models"):
        """Save trained models"""
        import joblib
        
        for name, model in self.models.items():
            filename = f"{filepath_prefix}_{name}.joblib"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        # Save feature information
        feature_info = {
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance
        }
        joblib.dump(feature_info, f"{filepath_prefix}_features.joblib")
        print(f"Saved feature information to {filepath_prefix}_features.joblib")

def main():
    """Main function to demonstrate improved model training"""
    print("🚀 IMPROVED STOCK PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Initialize improved predictor
    predictor = ImprovedStockPredictor()
    
    # Load your existing data (modify path as needed)
    try:
        # Load combined stock data
        print("📁 Loading data...")
        data = pd.read_csv("../processed_data/combined_stock_data.csv")
        print(f"✅ Loaded data with shape: {data.shape}")
        print(f"📊 Data columns: {list(data.columns)}")
        
        # Clean and prepare data
        print("\n🧹 Cleaning and preparing data...")
        data_clean = predictor.clean_and_prepare_data(data)
        
        # Create advanced features
        print("\n🔧 Creating advanced features...")
        data_with_features = predictor.create_advanced_features(data_clean)
        
        # Create target variables
        print("🎯 Creating target variables...")
        data_with_targets = predictor.create_target_variables(data_with_features)
        
        # Prepare features for training
        print("📊 Preparing features for training...")
        X, y, feature_cols = predictor.prepare_features(data_with_targets)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Train models
        print("\n🚀 Training improved models...")
        trained_models = predictor.train_models(X, y, optimize=True)
        
        # Evaluate model
        print("\n📈 Evaluating model performance...")
        # Split data for evaluation (use last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Retrain on training data
        for name, model in trained_models.items():
            if name != 'ensemble':
                model.fit(X_train, y_train)
        
        # Evaluate
        results = predictor.evaluate_model(X_test, y_test)
        
        # Plot feature importance
        print("\n📊 Plotting feature importance...")
        predictor.plot_feature_importance()
        
        # Save models
        print("\n💾 Saving improved models...")
        predictor.save_models()
        
        print("\n✅ Improved model training completed!")
        print(f"💡 Model achieved {results['direction_accuracy']:.1%} direction accuracy")
        print(f"   and {results['mape']:.1f}% MAPE on test data")
        
    except FileNotFoundError:
        print("❌ Data file not found. Please ensure the data path is correct.")
        print("💡 You can modify the data loading path in the main() function.")
        print("💡 Expected path: ../processed_data/combined_stock_data.csv")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
