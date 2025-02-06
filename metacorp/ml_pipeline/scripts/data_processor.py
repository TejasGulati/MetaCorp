from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os


class DataProcessor:
    """
    Data processor for company financial data.
    
    Column Descriptions and Units:
    - name: Company name (string)
    - industry: Industry/sector classification (string)
    - revenues: Annual revenue in millions USD
    - profits: Annual profit/income in millions USD
    - market_value: Market capitalization in millions USD
    - employees: Total number of employees (integer)
    - revenue_growth: Year-over-year revenue growth (percentage)
    - profit_margin: Profit as percentage of revenue (percentage)
    - employee_productivity: Revenue per employee (USD per employee)
    - market_to_revenue_ratio: Market value divided by revenue (ratio)

    Data Sources:
    1. Fortune 1000 Companies:
       - Revenues: Millions USD
       - Profits: Millions USD
       - Market Value: Millions USD
       - Employees: Headcount
       - Revenue Growth: Percentage change

    2. Tech Companies:
       - Original revenues in billions USD (converted to millions)
       - Original profits in billions USD (converted to millions)
       - Original market value in trillions USD (converted to millions)
       - Employees: Headcount
    """

    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_data_path = self.base_path / "data" / "processed"
        self.models_path = self.base_path / "data" / "models"
        
        # Create necessary directories
        for path in [self.raw_data_path, self.processed_data_path, self.models_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Define column units for documentation and validation
        self.column_units = {
            'name': 'string',
            'industry': 'string',
            'revenues': 'millions USD',
            'profits': 'millions USD',
            'market_value': 'millions USD',
            'employees': 'headcount',
            'revenue_growth': 'percentage',
            'profit_margin': 'percentage',
            'employee_productivity': 'USD per employee',
            'market_to_revenue_ratio': 'ratio'
        }

    
    def clean_monetary_value(self, value):
        """
        Helper function to clean monetary values
        Returns: float value in millions USD
        """
        if pd.isna(value) or str(value).strip() in ['-', '']:
            return 0
        try:
            # Remove currency symbols, commas, and handle parentheses for negative values
            cleaned = str(value).replace('$', '').replace(',', '')
            if '(' in cleaned and ')' in cleaned:
                cleaned = '-' + cleaned.replace('(', '').replace(')', '')
            return float(cleaned)
        except (ValueError, TypeError):
            return 0

    def clean_numeric_value(self, value):
        """
        Helper function to clean numeric values
        Returns: float value
        """
        if pd.isna(value) or str(value).strip() in ['-', '']:
            return 0
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return 0
    
    def load_and_combine_data(self):
        """
        Loads and combines company data from multiple sources.
        All monetary values are standardized to millions USD.
        """
        print("Loading CSV files...")
        # Load Fortune 1000 data (already in millions USD)
        fortune_df = pd.read_csv(self.raw_data_path / "Fortune_1000_Companies.csv")
        fortune_df.columns = fortune_df.columns.str.strip().str.lower()
        
        # Load Tech company data (needs conversion to millions USD)
        tech_df = pd.read_csv(self.raw_data_path / "Top_50_US_Tech_Companies.csv")
        tech_df.columns = tech_df.columns.str.strip()
        
        print("Processing Fortune 1000 data...")
        # Process Fortune 1000 data
        fortune_processed = pd.DataFrame()
        fortune_processed['name'] = fortune_df['name'].str.strip()
        
        # Handle industry/sector column
        if 'sector' in fortune_df.columns:
            fortune_processed['industry'] = fortune_df['sector'].str.strip()
        else:
            for possible_column in ['industry', 'business_sector', 'category']:
                if possible_column in fortune_df.columns:
                    fortune_processed['industry'] = fortune_df[possible_column].str.strip()
                    break
            else:
                fortune_processed['industry'] = 'Other'
        
        # Clean monetary and numeric values using helper functions (values already in millions USD)
        fortune_processed['revenues'] = fortune_df['revenues'].apply(self.clean_monetary_value)
        fortune_processed['profits'] = fortune_df['profits'].apply(self.clean_monetary_value)
        fortune_processed['market_value'] = fortune_df['market_value'].apply(self.clean_monetary_value)
        fortune_processed['employees'] = fortune_df['employees'].apply(self.clean_numeric_value)
        
        # Handle revenue growth (already in percentage)
        revenue_change_col = next((col for col in fortune_df.columns if 'revenue' in col.lower() and 'change' in col.lower()), None)
        if revenue_change_col:
            fortune_processed['revenue_growth'] = fortune_df[revenue_change_col].apply(
                lambda x: self.clean_numeric_value(str(x).rstrip('%')) if pd.notnull(x) else 0
            )
        else:
            fortune_processed['revenue_growth'] = 0
        
        print("Processing Tech company data...")
        # Process Tech company data with unit conversions
        tech_processed = pd.DataFrame()
        tech_processed['name'] = tech_df['Company Name'].str.strip()
        tech_processed['industry'] = 'Technology'
        
        # Convert from billions to millions USD
        tech_processed['revenues'] = tech_df['Annual Revenue 2022-2023 (USD in Billions)'].apply(
            lambda x: self.clean_numeric_value(x) * 1000  # Convert billions to millions
        )
        tech_processed['profits'] = tech_df['Annual Income Tax in 2022-2023 (USD in Billions)'].apply(
            lambda x: self.clean_numeric_value(x) * 1000  # Convert billions to millions
        )
        tech_processed['market_value'] = tech_df['Market Cap (USD in Trillions)'].apply(
            lambda x: self.clean_numeric_value(x) * 1000000  # Convert trillions to millions
        )
        tech_processed['employees'] = tech_df['Employee Size'].apply(self.clean_numeric_value)
        tech_processed['revenue_growth'] = 0  # placeholder as we don't have historical data (percentage)
        
        print("Calculating derived metrics...")
        # Calculate derived metrics for both datasets
        for df in [fortune_processed, tech_processed]:
            # Profit margin (percentage)
            df['profit_margin'] = np.where(
                df['revenues'] > 0,
                (df['profits'] / df['revenues']) * 100,
                0
            )
            
            # Employee productivity (USD per employee)
            df['employee_productivity'] = np.where(
                df['employees'] > 0,
                df['revenues'] / df['employees'],
                0
            )
            
            # Market to revenue ratio (ratio)
            df['market_to_revenue_ratio'] = np.where(
                df['revenues'] > 0,
                df['market_value'] / df['revenues'],
                0
            )
        
        print("Combining datasets...")
        # Combine datasets
        combined_df = pd.concat([fortune_processed, tech_processed], ignore_index=True)
        
        # Remove rows with invalid data
        combined_df = combined_df[
            (combined_df['revenues'] > 0) & 
            (combined_df['employees'] > 0)
        ]
        
        print(f"Final dataset shape: {combined_df.shape}")
        print(f"Number of Fortune 1000 companies: {len(fortune_processed)}")
        print(f"Number of Tech companies: {len(tech_processed)}")
        print("\nColumn Units:")
        for col, unit in self.column_units.items():
            print(f"- {col}: {unit}")
        
        # Save processed data
        combined_df.to_csv(self.processed_data_path / "combined_company_data.csv", index=False)
        return combined_df
    
class ModelTrainer:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path(__file__).parent.parent
        
        self.processed_data_path = base_path / "data" / "processed"
        self.models_path = base_path / "data" / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.cache_files = {
            'scaler': self.models_path / 'scaler.joblib',
            'feature_names': self.models_path / 'feature_names.joblib',
            'market_value': self.models_path / 'market_value_model.joblib',
            'profit_margin': self.models_path / 'profit_margin_model.joblib',
            'revenue_growth': self.models_path / 'revenue_growth_model.joblib',
            'evaluation_results': self.models_path / 'evaluation_results.joblib'
        }
        
        # Further optimized parameters based on feature importance analysis
        self.model_params = {
            'market_value': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 3500,    # Increased for better market value prediction
                    'max_depth': 14,         # Increased to capture complex market relationships
                    'min_samples_split': 8,  # Increased to prevent overfitting
                    'min_samples_leaf': 4,   # Increased for better generalization
                    'learning_rate': 0.002,  # Reduced for finer convergence
                    'subsample': 0.92,       # Increased for better stability
                    'loss': 'huber',         # Changed to huber for robustness
                    'alpha': 0.95,           # Huber loss parameter
                    'random_state': 42,
                    'warm_start': True,
                    'validation_fraction': 0.1
                }
            },
            'profit_margin': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 1000,    # Increased based on validation curves
                    'max_depth': 10,         # Increased for complex profit patterns
                    'min_samples_split': 5,
                    'min_samples_leaf': 3,
                    'learning_rate': 0.005,
                    'subsample': 0.88,
                    'loss': 'huber',         # Changed to huber for robustness
                    'alpha': 0.9,            # Huber loss parameter
                    'random_state': 42,
                    'warm_start': True,
                    'validation_fraction': 0.1
                }
            },
            'revenue_growth': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 800,     # Adjusted based on convergence analysis
                    'max_depth': 8,
                    'min_samples_split': 7,
                    'min_samples_leaf': 4,
                    'learning_rate': 0.008,
                    'subsample': 0.85,
                    'loss': 'huber',         # Changed to huber for robustness
                    'alpha': 0.9,            # Huber loss parameter
                    'random_state': 42,
                    'warm_start': True,
                    'validation_fraction': 0.1
                }
            }
        }

    def _check_cached_models(self):
        """Check if all required model files exist in cache"""
        return all(cache_file.exists() for cache_file in self.cache_files.values())

    def _load_cached_models(self):
        """Load models and related data from cache"""
        try:
            print("Loading models from cache...")
            models = {}
            for target in ['market_value', 'profit_margin', 'revenue_growth']:
                models[target] = joblib.load(self.cache_files[target])
            
            evaluation_results = joblib.load(self.cache_files['evaluation_results'])
            scaler = joblib.load(self.cache_files['scaler'])
            feature_names = joblib.load(self.cache_files['feature_names'])
            
            print("Successfully loaded all models and data from cache!")
            return models, evaluation_results, scaler, feature_names
        except Exception as e:
            print(f"Error loading cached models: {str(e)}")
            return None, None, None, None

    def _save_models(self, models, evaluation_results, scaler, feature_names):
        """Save models and related data to cache"""
        try:
            print("\nSaving models and data to cache...")
            for target, model in models.items():
                joblib.dump(model, self.cache_files[target])
            
            joblib.dump(evaluation_results, self.cache_files['evaluation_results'])
            joblib.dump(scaler, self.cache_files['scaler'])
            joblib.dump(feature_names, self.cache_files['feature_names'])
            
            print("Successfully saved all models and data to cache!")
        except Exception as e:
            print(f"Error saving models to cache: {str(e)}")

    def _engineer_features(self, df):
        """Enhanced feature engineering with focus on high-impact features"""
        df = df.copy()
        eps = 1e-10
        
        # Core financial ratios with refined weights
        df['profit_growth'] = np.where(df['revenues'] > eps,
                                     df['profits'] / df['revenues'] * 100,
                                     0)
        
        # Enhanced employee metrics
        df['revenue_per_employee'] = np.where(df['employees'] > eps,
                                            df['revenues'] / df['employees'],
                                            0)
        df['profit_per_employee'] = np.where(df['employees'] > eps,
                                           df['profits'] / df['employees'],
                                           0)
        df['market_value_per_employee'] = np.where(df['employees'] > eps,
                                                  df['market_value'] / df['employees'],
                                                  0)
        
        # Expanded industry metrics
        metrics = ['market_value', 'revenues', 'profits', 'employees']
        for metric in metrics:
            industry_avg = df.groupby('industry')[metric].transform('mean')
            industry_std = df.groupby('industry')[metric].transform('std')
            industry_median = df.groupby('industry')[metric].transform('median')
            
            df[f'industry_{metric}_zscore'] = np.where(
                industry_std > eps,
                (df[metric] - industry_avg) / industry_std,
                0
            )
            df[f'industry_{metric}_rel_median'] = np.where(
                industry_median > eps,
                df[metric] / industry_median,
                0
            )
        
        # Enhanced composite scores with dynamic weights
        weights = {
            'revenues': 0.35,
            'employees': 0.25,
            'market_value': 0.4
        }
        
        max_vals = {
            metric: df[metric].max() if df[metric].max() > eps else 1
            for metric in weights.keys()
        }
        
        df['size_score'] = sum(
            (df[metric] / max_vals[metric]) * weight
            for metric, weight in weights.items()
        )
        
        # Enhanced efficiency metrics
        max_profit_margin = max(df['profit_margin'].max(), eps)
        max_revenue_growth = max(df['revenue_growth'].max(), eps)
        max_productivity = max(df['revenue_per_employee'].max(), eps)
        
        df['efficiency_score'] = (
            (df['profit_margin'] / max_profit_margin) * 0.5 +
            (df['revenue_growth'] / max_revenue_growth) * 0.3 +
            (df['revenue_per_employee'] / max_productivity) * 0.2
        )
        
        # Advanced growth metrics
        df['composite_growth'] = (
            df['revenue_growth'] * 0.6 +
            df['profit_growth'] * 0.3 +
            (df['market_value_per_employee'].pct_change().fillna(0)) * 0.1
        )
        
        # Interaction features
        df['size_efficiency_interaction'] = df['size_score'] * df['efficiency_score']
        df['growth_efficiency_interaction'] = df['composite_growth'] * df['efficiency_score']
        
        # Volatility measures
        for metric in ['profit_margin', 'revenue_growth']:
            rolling_std = df.groupby('industry')[metric].transform(
                lambda x: x.rolling(2, min_periods=1).std()
            ).fillna(0)
            df[f'{metric}_volatility'] = rolling_std
        
        return df.replace([np.inf, -np.inf], 0)

    def _handle_outliers(self, df, numerical_features):
        """Enhanced outlier handling with feature-specific thresholds"""
        df = df.copy()
        
        # Define feature groups for different threshold treatments
        threshold_groups = {
            'market': ['market_value', 'market_value_per_employee', 'industry_market_value_zscore'],
            'growth': ['revenue_growth', 'profit_growth', 'composite_growth'],
            'efficiency': ['profit_margin', 'efficiency_score', 'revenue_per_employee'],
            'ratio': ['size_efficiency_interaction', 'growth_efficiency_interaction'],
            'default': []  # All other features
        }
        
        thresholds = {
            'market': 12.0,    # Increased for market features
            'growth': 4.5,     # Reduced for growth metrics
            'efficiency': 5.5, # Moderate for efficiency metrics
            'ratio': 6.0,      # Standard for ratios
            'default': 6.0     # Default threshold
        }
        
        # Create reverse mapping of features to their groups
        feature_to_group = {}
        for group, features in threshold_groups.items():
            for feature in features:
                feature_to_group[feature] = group
        
        for feature in numerical_features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            # Get appropriate threshold
            group = feature_to_group.get(feature, 'default')
            threshold = thresholds[group]
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Apply clipping with additional logging
            original_outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
            df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
            
            if original_outliers > 0:
                print(f"Handled {original_outliers} outliers in {feature}")
        
        return df

    def prepare_training_data(self, df):
        """Optimized data preparation with enhanced feature selection"""
        try:
            print("Starting data preparation...")
            df = self._engineer_features(df)
            
            # Expanded feature set based on importance analysis
            numerical_features = [
                'revenues', 'profits', 'employees',
                'revenue_growth', 'profit_margin', 'revenue_per_employee',
                'profit_per_employee', 'market_value_per_employee',
                'profit_growth', 'efficiency_score', 'size_score',
                'industry_market_value_zscore', 'industry_revenues_zscore',
                'industry_profits_zscore', 'industry_employees_zscore',
                'industry_market_value_rel_median', 'industry_revenues_rel_median',
                'composite_growth', 'size_efficiency_interaction',
                'growth_efficiency_interaction',
                'profit_margin_volatility', 'revenue_growth_volatility'
            ]
            
            categorical_features = ['industry']
            target_variables = ['market_value', 'profit_margin', 'revenue_growth']
            
            print("Handling outliers...")
            df = self._handle_outliers(df, numerical_features)
            
            X = df[numerical_features + categorical_features].copy()
            
            # Enhanced scaling with feature-specific treatment
            print("Applying robust scaling...")
            scaler = RobustScaler(quantile_range=(5, 95))  # Adjusted quantile range
            X[numerical_features] = scaler.fit_transform(X[numerical_features])
            joblib.dump(scaler, self.models_path / 'scaler.joblib')
            
            # One-hot encoding with improved handling
            print("Performing one-hot encoding...")
            X = pd.get_dummies(X, columns=['industry'], drop_first=True, sparse=False)
            joblib.dump(list(X.columns), self.models_path / 'feature_names.joblib')
            
            # Enhanced target processing
            print("Processing target variables...")
            y = df[target_variables].copy()
            
            # Log transform for market_value with offset
            offset = np.abs(y['market_value'].min()) + 1 if y['market_value'].min() < 0 else 0
            y['market_value'] = np.where(y['market_value'] + offset > 0,
                                       np.log1p(y['market_value'] + offset),
                                       0)
            
            # Stratified splitting with enhanced binning
            print("Performing stratified splitting...")
            splits = {}
            for target in target_variables:
                # Dynamic bin calculation based on data distribution
                n_bins = min(10, len(y[target].unique()))
                bins = pd.qcut(y[target], q=n_bins, labels=False, duplicates='drop')
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y[target],
                    test_size=0.2,
                    random_state=42,
                    stratify=bins
                )
                
                splits[target] = (X_train, X_test, y_train, y_test)
            
            print("Data preparation completed successfully!")
            return splits
        
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            raise

    def train_and_evaluate_models(self, splits, force_retrain=False):
        """Enhanced training and evaluation with improved metrics, validation, and caching"""
        if not force_retrain and self._check_cached_models():
            # Load cached models if available
            models, evaluation_results, _, _ = self._load_cached_models()
            if models and evaluation_results:
                print("\nUsing cached models - skipping training")
                return models, evaluation_results

        print("\nNo cached models found or force_retrain=True. Starting model training...")
        evaluation_results = {}
        models = {}
        
        for target, (X_train, X_test, y_train, y_test) in splits.items():
            try:
                print(f"\nTraining model for: {target}")
                
                model_config = self.model_params[target]
                model = model_config['model_type'](**model_config['params'])
                
                # Enhanced cross-validation
                cv = KFold(n_splits=7, shuffle=True, random_state=42)
                
                # Compute multiple CV metrics
                # Compute multiple CV metrics
                cv_metrics = {
                    'r2': cross_val_score(model, X_train, y_train, cv=cv, 
                                        scoring='r2', n_jobs=-1),
                    'neg_mae': cross_val_score(model, X_train, y_train, cv=cv, 
                                             scoring='neg_mean_absolute_error', n_jobs=-1),
                    'neg_rmse': cross_val_score(model, X_train, y_train, cv=cv, 
                                              scoring='neg_root_mean_squared_error', n_jobs=-1)
                }
                
                print(f"Training final model for {target}...")
                model.fit(X_train, y_train)
                models[target] = model
                
                # Predictions and transformations
                y_pred = model.predict(X_test)
                y_pred_train = model.predict(X_train)
                
                # Handle market_value inverse transform
                if target == 'market_value':
                    offset = np.abs(y_test.min()) + 1 if y_test.min() < 0 else 0
                    y_test_orig = np.expm1(y_test) - offset
                    y_pred_orig = np.expm1(y_pred) - offset
                    y_train_orig = np.expm1(y_train) - offset
                    y_pred_train_orig = np.expm1(y_pred_train) - offset
                else:
                    y_test_orig = y_test
                    y_pred_orig = y_pred
                    y_train_orig = y_train
                    y_pred_train_orig = y_pred_train
                
                # Comprehensive metrics
                metrics = {
                    'R2_Score_Test': r2_score(y_test_orig, y_pred_orig),
                    'R2_Score_Train': r2_score(y_train_orig, y_pred_train_orig),
                    'MAE': mean_absolute_error(y_test_orig, y_pred_orig),
                    'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
                    'CV_R2_Mean': cv_metrics['r2'].mean(),
                    'CV_R2_Std': cv_metrics['r2'].std(),
                    'CV_MAE_Mean': -cv_metrics['neg_mae'].mean(),
                    'CV_RMSE_Mean': -cv_metrics['neg_rmse'].mean(),
                    'Feature_Importance': dict(sorted(
                        zip(X_train.columns, model.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    ))
                }
                
                evaluation_results[target] = metrics
                
                # Print detailed metrics
                self._print_metrics(target, metrics)
                
            except Exception as e:
                print(f"Error training model for {target}: {str(e)}")
                continue
        
        # Save models and related data to cache
        scaler = joblib.load(self.models_path / 'scaler.joblib')
        feature_names = joblib.load(self.models_path / 'feature_names.joblib')
        self._save_models(models, evaluation_results, scaler, feature_names)
        
        return models, evaluation_results
    
    def _print_metrics(self, target, metrics):
        """Print formatted metrics"""
        print(f"\nDetailed Metrics for {target}:")
        print(f"Test R² Score: {metrics['R2_Score_Test']:.4f}")
        print(f"Training R² Score: {metrics['R2_Score_Train']:.4f}")
        print(f"Cross-validation R² (mean ± std): {metrics['CV_R2_Mean']:.4f} ± {metrics['CV_R2_Std']:.4f}")
        print(f"Mean Absolute Error: {metrics['MAE']:.4f}")
        print(f"Root Mean Squared Error: {metrics['RMSE']:.4f}")
        
        print("\nTop 5 Most Important Features:")
        sorted_features = dict(sorted(metrics['Feature_Importance'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5])
        for feature, importance in sorted_features.items():
            print(f"{feature}: {importance:.4f}")

# Usage script
if __name__ == "__main__":
    try:
        # Initialize processor and trainer
        processor = DataProcessor()
        trainer = ModelTrainer(processor.base_path)
        
        # Check if we have cached models
        if trainer._check_cached_models():
            print("Found cached models! Loading previous training results...")
            models, evaluation_results, _, _ = trainer._load_cached_models()
            
            # Print cached model metrics
            for target, metrics in evaluation_results.items():
                trainer._print_metrics(target, metrics)
                
        else:
            print("No cached models found. Processing data and training new models...")
            # Load and process data
            combined_data = processor.load_and_combine_data()
            
            # Prepare data with train-test split
            splits = trainer.prepare_training_data(combined_data)
            
            # Train and evaluate models
            models, evaluation_results = trainer.train_and_evaluate_models(splits)
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise