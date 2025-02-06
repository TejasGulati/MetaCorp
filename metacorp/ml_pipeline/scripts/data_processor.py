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
    Enhanced data processor for company financial data.
    
    Column Descriptions and Units:
    - name: Company name (string)
    - industry/sector: Industry/sector classification (string)
    - revenues: Annual revenue in millions USD
    - profits: Annual profit/income in millions USD
    - market_value: Market capitalization in millions USD
    - employees: Total number of employees (integer)
    - revenue_growth: Year-over-year revenue growth (percentage)
    - profit_margin: Profit as percentage of revenue (percentage)
    - employee_productivity: Revenue per employee (USD per employee)
    - market_to_revenue_ratio: Market value divided by revenue (ratio)
    - rank: Fortune 1000 ranking (integer)
    - rank_change: Change in ranking from previous year (integer)
    - ceo_founder: CEO is company founder (boolean)
    - ceo_woman: CEO is woman (boolean)
    - profitable: Company is profitable (boolean)
    - market_cap: Market capitalization (millions USD)
    """

    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_data_path = self.base_path / "data" / "processed"
        self.models_path = self.base_path / "data" / "models"
        
        # Create necessary directories
        for path in [self.raw_data_path, self.processed_data_path, self.models_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Enhanced column units with new metrics
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
            'market_to_revenue_ratio': 'ratio',
            'rank': 'integer',
            'rank_change': 'integer',
            'ceo_founder': 'boolean',
            'ceo_woman': 'boolean',
            'profitable': 'boolean',
            'market_cap': 'millions USD'
        }

    def clean_monetary_value(self, value):
        """Enhanced helper function to clean monetary values"""
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
        """Enhanced helper function to clean numeric values"""
        if pd.isna(value) or str(value).strip() in ['-', '']:
            return 0
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return 0
    
    def clean_boolean_value(self, value):
        """Helper function to clean boolean values"""
        if pd.isna(value):
            return False
        return str(value).lower() in ['yes', 'true', '1', 'y']
    
    def load_and_combine_data(self):
        """
        Enhanced data loading function incorporating Fortune 1000 2022 dataset
        with proper column name handling
        """
        print("Loading CSV files...")
        try:
            # Load all datasets
            fortune_df = pd.read_csv(self.raw_data_path / "Fortune_1000_Companies.csv")
            tech_df = pd.read_csv(self.raw_data_path / "Top_50_US_Tech_Companies.csv")
            fortune_2022_df = pd.read_csv(self.raw_data_path / "Fortune_1000_2022.csv")
            
            # Print column names for debugging
            print("\nOriginal column names:")
            print("Fortune 1000:", fortune_df.columns.tolist())
            print("Tech Companies:", tech_df.columns.tolist())
            print("Fortune 2022:", fortune_2022_df.columns.tolist())
            
            # Standardize column names
            for df in [fortune_df, tech_df, fortune_2022_df]:
                df.columns = df.columns.str.strip().str.lower()
            
            print("\nStandardized column names:")
            print("Fortune 1000:", fortune_df.columns.tolist())
            print("Tech Companies:", tech_df.columns.tolist())
            print("Fortune 2022:", fortune_2022_df.columns.tolist())
            
            print("\nProcessing Fortune 1000 2022 data...")
            # Process Fortune 2022 data
            fortune_2022_processed = pd.DataFrame()
            fortune_2022_processed['name'] = fortune_2022_df['company'].str.strip()
            fortune_2022_processed['industry'] = fortune_2022_df['sector'].str.strip()
            fortune_2022_processed['revenues'] = fortune_2022_df['revenue'].apply(self.clean_monetary_value)
            fortune_2022_processed['profits'] = fortune_2022_df['profit'].apply(self.clean_monetary_value)
            fortune_2022_processed['employees'] = fortune_2022_df['num. of employees'].apply(self.clean_numeric_value)
            
            # Handle market cap if present
            if 'market cap' in fortune_2022_df.columns:
                fortune_2022_processed['market_value'] = fortune_2022_df['market cap'].apply(self.clean_monetary_value)
            else:
                fortune_2022_processed['market_value'] = 0
            
            # New features from 2022 dataset
            fortune_2022_processed['rank'] = fortune_2022_df['rank'].apply(self.clean_numeric_value)
            fortune_2022_processed['rank_change'] = fortune_2022_df['rank_change'].apply(self.clean_numeric_value)
            fortune_2022_processed['ceo_founder'] = fortune_2022_df['ceo_founder'].apply(self.clean_boolean_value)
            fortune_2022_processed['ceo_woman'] = fortune_2022_df['ceo_woman'].apply(self.clean_boolean_value)
            fortune_2022_processed['profitable'] = fortune_2022_df['profitable'].apply(self.clean_boolean_value)
            
            print("\nProcessing original Fortune 1000 data...")
            # Process original Fortune data
            fortune_processed = pd.DataFrame()
            # Handle name column variations
            name_col = next(col for col in fortune_df.columns if 'name' in col)
            fortune_processed['name'] = fortune_df[name_col].str.strip()
            
            # Handle sector/industry column
            sector_col = next((col for col in fortune_df.columns if col in ['sector', 'industry']), None)
            fortune_processed['industry'] = fortune_df[sector_col].str.strip() if sector_col else 'Other'
            
            # Handle revenue column variations
            revenue_col = next(col for col in fortune_df.columns if 'revenues' in col)
            fortune_processed['revenues'] = fortune_df[revenue_col].apply(self.clean_monetary_value)
            
            # Handle profits column variations
            profits_col = next(col for col in fortune_df.columns if 'profits' in col)
            fortune_processed['profits'] = fortune_df[profits_col].apply(self.clean_monetary_value)
            
            # Handle market value column variations
            market_col = next(col for col in fortune_df.columns if 'market' in col)
            fortune_processed['market_value'] = fortune_df[market_col].apply(self.clean_monetary_value)
            
            # Handle employees column variations
            employees_col = next(col for col in fortune_df.columns if 'employees' in col)
            fortune_processed['employees'] = fortune_df[employees_col].apply(self.clean_numeric_value)
            
            # Add placeholder columns for new features
            fortune_processed['rank'] = fortune_df['rank'].apply(self.clean_numeric_value)
            fortune_processed['rank_change'] = 0
            fortune_processed['ceo_founder'] = False
            fortune_processed['ceo_woman'] = False
            fortune_processed['profitable'] = fortune_processed['profits'] > 0
            
            print("\nProcessing Tech company data...")
            # Process Tech company data
            tech_processed = pd.DataFrame()
            # Handle company name variations
            company_col = next(col for col in tech_df.columns if 'company' in col.lower())
            tech_processed['name'] = tech_df[company_col].str.strip()
            
            tech_processed['industry'] = 'Technology'
            
            # Handle revenue column variations
            revenue_col = next(col for col in tech_df.columns if 'revenue' in col.lower())
            tech_processed['revenues'] = tech_df[revenue_col].apply(
                lambda x: self.clean_numeric_value(x) * 1000  # Convert billions to millions
            )
            
            # Handle income/profit column variations
            income_col = next(col for col in tech_df.columns if 'income' in col.lower())
            tech_processed['profits'] = tech_df[income_col].apply(
                lambda x: self.clean_numeric_value(x) * 1000  # Convert billions to millions
            )
            
            # Handle market cap column variations
            market_col = next(col for col in tech_df.columns if 'market' in col.lower())
            tech_processed['market_value'] = tech_df[market_col].apply(
                lambda x: self.clean_numeric_value(x) * 1000000  # Convert trillions to millions
            )
            
            # Handle employee column variations
            employee_col = next(col for col in tech_df.columns if 'employee' in col.lower())
            tech_processed['employees'] = tech_df[employee_col].apply(self.clean_numeric_value)
            
            # Add placeholder columns for new features
            tech_processed['rank'] = 0
            tech_processed['rank_change'] = 0
            tech_processed['ceo_founder'] = False
            tech_processed['ceo_woman'] = False
            tech_processed['profitable'] = tech_processed['profits'] > 0
            
            print("\nCombining datasets...")
            # Combine all datasets
            dataframes = [fortune_2022_processed, fortune_processed, tech_processed]
            
            # Ensure all required columns exist in each dataframe
            required_columns = ['name', 'industry', 'revenues', 'profits', 'market_value', 
                              'employees', 'rank', 'rank_change', 'ceo_founder', 'ceo_woman', 
                              'profitable']
            
            for df in dataframes:
                for col in required_columns:
                    if col not in df.columns:
                        if col in ['ceo_founder', 'ceo_woman', 'profitable']:
                            df[col] = False
                        else:
                            df[col] = 0
            
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            print("\nCalculating derived metrics...")
            # Calculate derived metrics
            combined_df['profit_margin'] = np.where(
                combined_df['revenues'] > 0,
                (combined_df['profits'] / combined_df['revenues']) * 100,
                0
            )
            
            combined_df['employee_productivity'] = np.where(
                combined_df['employees'] > 0,
                combined_df['revenues'] / combined_df['employees'],
                0
            )
            
            combined_df['market_to_revenue_ratio'] = np.where(
                combined_df['revenues'] > 0,
                combined_df['market_value'] / combined_df['revenues'],
                0
            )
            
            # Remove duplicates based on name
            combined_df = combined_df.drop_duplicates(subset=['name'], keep='first')
            
            # Remove rows with invalid data
            combined_df = combined_df[
                (combined_df['revenues'] > 0) & 
                (combined_df['employees'] > 0)
            ]
            
            print(f"\nFinal dataset shape: {combined_df.shape}")
            print("\nColumn Units:")
            for col, unit in self.column_units.items():
                if col in combined_df.columns:
                    print(f"- {col}: {unit}")
            
            # Save processed data
            combined_df.to_csv(self.processed_data_path / "combined_company_data.csv", index=False)
            return combined_df
            
        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            raise

class ModelTrainer:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path(__file__).parent.parent
        
        self.processed_data_path = base_path / "data" / "processed"
        self.models_path = base_path / "data" / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Define targets
        self.target_variables = ['market_value', 'profit_margin', 'revenue_growth']
        
        # Reorganized cache files structure by target
        self.cache_files = {}
        for target in self.target_variables:
            self.cache_files[target] = {
                'model': self.models_path / f'{target}_model.joblib',
                'scaler': self.models_path / f'{target}_scaler.joblib',
                'feature_names': self.models_path / f'{target}_feature_names.joblib'
            }
        # Global evaluation results cache
        self.cache_files['evaluation_results'] = self.models_path / 'evaluation_results.joblib'
        
        # Optimized model parameters with adjusted parameters for revenue_growth
        self.model_params = {
            'market_value': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 3000,
                    'max_depth': 12,
                    'min_samples_split': 10,
                    'min_samples_leaf': 6,
                    'learning_rate': 0.002,
                    'subsample': 0.8,
                    'loss': 'huber',
                    'alpha': 0.95,
                    'random_state': 42
                }
            },
            'profit_margin': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 1000,
                    'max_depth': 8,
                    'min_samples_split': 8,
                    'min_samples_leaf': 4,
                    'learning_rate': 0.005,
                    'subsample': 0.85,
                    'loss': 'huber',
                    'alpha': 0.92,
                    'random_state': 42
                }
            },
            'revenue_growth': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 1200,  # Increased for better learning
                    'max_depth': 6,        # Reduced to prevent overfitting
                    'min_samples_split': 5,
                    'min_samples_leaf': 3,
                    'learning_rate': 0.01,
                    'subsample': 0.9,
                    'loss': 'huber',
                    'alpha': 0.9,
                    'random_state': 42
                }
            }
        }

    def _calculate_revenue_growth(self, df):
        """
        Calculate revenue growth using historical data and industry comparisons
        """
        # Sort by company and year (assuming year information is available)
        df = df.sort_values(['name', 'year']) if 'year' in df.columns else df.copy()
        
        # Calculate YoY revenue growth where possible
        if 'year' in df.columns:
            df['revenue_growth'] = df.groupby('name')['revenues'].pct_change() * 100
        else:
            # If no year data, estimate growth using industry comparisons
            industry_median_revenue = df.groupby('industry')['revenues'].transform('median')
            industry_mean_revenue = df.groupby('industry')['revenues'].transform('mean')
            
            # Calculate relative position to industry
            df['revenue_growth'] = ((df['revenues'] - industry_median_revenue) / 
                                  (industry_mean_revenue + 1e-10)) * 10
        
        # Handle infinities and NANs
        df['revenue_growth'] = df['revenue_growth'].replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with industry medians
        industry_median_growth = df.groupby('industry')['revenue_growth'].transform('median')
        df['revenue_growth'] = df['revenue_growth'].fillna(industry_median_growth)
        
        # If still any NANs, fill with global median
        global_median = df['revenue_growth'].median()
        df['revenue_growth'] = df['revenue_growth'].fillna(global_median)
        
        # Clip extreme values
        lower_bound = df['revenue_growth'].quantile(0.05)
        upper_bound = df['revenue_growth'].quantile(0.95)
        df['revenue_growth'] = df['revenue_growth'].clip(lower_bound, upper_bound)
        
        return df

    def _engineer_features(self, df):
        """Enhanced feature engineering with proper revenue growth calculation"""
        df = df.copy()
        
        # Calculate revenue growth first
        df = self._calculate_revenue_growth(df)
        
        # Rest of feature engineering
        eps = 1e-10

        # Core financial ratios
        df['profit_growth'] = np.where(df['revenues'] > eps,
                                    df['profits'] / df['revenues'] * 100,
                                    0)

        # Employee metrics
        df['revenue_per_employee'] = np.where(df['employees'] > eps,
                                        df['revenues'] / df['employees'],
                                        0)
        df['profit_per_employee'] = np.where(df['employees'] > eps,
                                       df['profits'] / df['employees'],
                                       0)
        df['market_value_per_employee'] = np.where(df['employees'] > eps,
                                              df['market_value'] / df['employees'],
                                              0)

        # Industry metrics
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

        # Composite scores
        max_vals = {metric: max(df[metric].max(), eps) for metric in metrics}
        
        df['size_score'] = (
            (df['revenues'] / max_vals['revenues']) * 0.35 +
            (df['employees'] / max_vals['employees']) * 0.25 +
            (df['market_value'] / max_vals['market_value']) * 0.4
        )

        # Efficiency metrics
        max_profit_margin = max(df['profit_margin'].max(), eps)
        max_revenue_per_employee = max(df['revenue_per_employee'].max(), eps)

        df['efficiency_score'] = (
            (df['profit_margin'] / max_profit_margin) * 0.5 +
            (df['revenue_per_employee'] / max_revenue_per_employee) * 0.5
        )

        # Growth metrics
        df['composite_growth'] = (
            df['revenue_growth'] * 0.4 +
            df['profit_growth'] * 0.6
        )

        # Interaction features
        df['size_efficiency_interaction'] = df['size_score'] * df['efficiency_score']
        df['growth_efficiency_interaction'] = df['composite_growth'] * df['efficiency_score']

        # Market sentiment indicators
        df['market_confidence'] = np.where(
            df['revenues'] > eps,
            df['market_value'] / df['revenues'],
            0
        )

        # Volatility measures
        for metric in ['profit_margin', 'revenue_growth']:
            rolling_std = df.groupby('industry')[metric].transform(
                lambda x: x.rolling(2, min_periods=1).std()
            ).fillna(0)
            df[f'{metric}_volatility'] = rolling_std

        return df.replace([np.inf, -np.inf], 0)

    def _check_cached_models(self):
        """Check if all required model files exist in cache"""
        all_files_exist = True
        for target in self.target_variables:
            target_files_exist = all(
                self.cache_files[target][file_type].exists() 
                for file_type in ['model', 'scaler', 'feature_names']
            )
            if not target_files_exist:
                all_files_exist = False
                break
        
        return all_files_exist and self.cache_files['evaluation_results'].exists()

    def _load_cached_models(self):
        """Load models and related data from cache"""
        try:
            print("Loading models from cache...")
            models = {}
            scalers = {}
            feature_names = {}
            
            for target in self.target_variables:
                models[target] = joblib.load(self.cache_files[target]['model'])
                scalers[target] = joblib.load(self.cache_files[target]['scaler'])
                feature_names[target] = joblib.load(self.cache_files[target]['feature_names'])
            
            evaluation_results = joblib.load(self.cache_files['evaluation_results'])
            
            print("Successfully loaded all models and data from cache!")
            return models, evaluation_results, scalers, feature_names
        except Exception as e:
            print(f"Error loading cached models: {str(e)}")
            return None, None, None, None

    def _save_models(self, models, evaluation_results, scalers, feature_names):
        """Save models and related data to cache"""
        try:
            print("\nSaving models and data to cache...")
            for target in self.target_variables:
                joblib.dump(models[target], self.cache_files[target]['model'])
                joblib.dump(scalers[target], self.cache_files[target]['scaler'])
                joblib.dump(feature_names[target], self.cache_files[target]['feature_names'])
            
            joblib.dump(evaluation_results, self.cache_files['evaluation_results'])
            print("Successfully saved all models and data to cache!")
        except Exception as e:
            print(f"Error saving models to cache: {str(e)}")

    def prepare_training_data(self, df):
        """Prepare training data with enhanced feature engineering and scaling"""
        try:
            print("Starting data preparation...")
            df = self._engineer_features(df)
            
            # Base feature set
            base_numerical_features = [
                'revenues', 'profits', 'employees',
                'revenue_per_employee', 'profit_per_employee', 
                'market_value_per_employee', 'profit_growth', 
                'efficiency_score', 'size_score',
                'industry_market_value_zscore', 'industry_revenues_zscore',
                'industry_profits_zscore', 'industry_employees_zscore',
                'industry_market_value_rel_median', 'industry_revenues_rel_median',
                'composite_growth', 'size_efficiency_interaction',
                'growth_efficiency_interaction',
                'profit_margin_volatility', 'revenue_growth_volatility'
            ]
            
            categorical_features = ['industry']
            
            # Create target-specific feature sets
            target_features = {
                'market_value': base_numerical_features + ['profit_margin', 'revenue_growth'],
                'profit_margin': base_numerical_features + ['market_value', 'revenue_growth'],
                'revenue_growth': base_numerical_features + ['market_value', 'profit_margin']
            }
            
            print("Handling outliers...")
            df = self._handle_outliers(df, base_numerical_features)
            
            splits = {}
            scalers = {}
            feature_names = {}
            
            for target in self.target_variables:
                print(f"\nPreparing data for {target} model...")
                current_features = target_features[target]
                X = df[current_features + categorical_features].copy()
                
                # Scale numerical features
                print(f"Applying robust scaling...")
                scaler = RobustScaler(quantile_range=(5, 95))
                X[current_features] = scaler.fit_transform(X[current_features])
                scalers[target] = scaler
                
                # One-hot encoding
                X = pd.get_dummies(X, columns=['industry'], drop_first=True, sparse=False)
                feature_names[target] = list(X.columns)
                
                # Process target variable
                y = df[target].copy()
                if target == 'market_value':
                    offset = np.abs(y.min()) + 1 if y.min() < 0 else 0
                    y = np.where(y + offset > 0, np.log1p(y + offset), 0)
                
                # Convert to numpy array if not already
                y_array = y.values if isinstance(y, pd.Series) else y
                
                # Create bins for stratification
                unique_values = np.unique(y_array)
                n_bins = min(10, len(unique_values))
                
                if n_bins > 1:
                    bins = pd.qcut(pd.Series(y_array), q=n_bins, labels=False, duplicates='drop')
                else:
                    bins = None
                
                # Perform the split with optional stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_array,
                    test_size=0.2,
                    random_state=42,
                    stratify=bins
                )
                
                splits[target] = (X_train, X_test, y_train, y_test)
            
            print("Data preparation completed successfully!")
            return splits, scalers, feature_names
        
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            raise

    def train_and_evaluate_models(self, splits, scalers, feature_names, force_retrain=False):
        """Train and evaluate models with comprehensive metrics and proper caching"""
        if not force_retrain and self._check_cached_models():
            # Load cached models if available
            return self._load_cached_models()

        print("\nNo cached models found or force_retrain=True. Starting model training...")
        evaluation_results = {}
        models = {}
        
        for target in self.target_variables:
            try:
                X_train, X_test, y_train, y_test = splits[target]
                print(f"\nTraining model for: {target}")
                
                model_config = self.model_params[target]
                model = model_config['model_type'](**model_config['params'])
                
                # Enhanced cross-validation
                cv = KFold(n_splits=7, shuffle=True, random_state=42)
                
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
                self._print_metrics(target, metrics)
                
            except Exception as e:
                print(f"Error training model for {target}: {str(e)}")
                continue
        
        # Save models and related data to cache
        self._save_models(models, evaluation_results, scalers, feature_names)
        
        return models, evaluation_results, scalers, feature_names
    
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

    def predict(self, input_data, target='market_value'):
        """
        Make predictions using trained models
        """
        try:
            # Load cached models if available
            models, _, scalers, feature_names = self._load_cached_models()
            
            if not models or target not in models:
                raise ValueError(f"No trained model found for target: {target}")
            
            # Prepare input data
            processed_data = self._engineer_features(input_data.copy())
            
            # Get relevant features
            X = processed_data[feature_names[target]]
            
            # Scale features
            X_scaled = scalers[target].transform(X)
            
            # Make prediction
            prediction = models[target].predict(X_scaled)
            
            # Inverse transform for market_value
            if target == 'market_value':
                prediction = np.expm1(prediction)
            
            return prediction
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

    def _handle_outliers(self, df, numerical_features):
        """Handle outliers with feature-specific thresholds"""
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
            try:
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
            except KeyError as e:
                print(f"Error handling outliers for feature {feature}: {str(e)}")
                continue
        
        return df
    
if __name__ == "__main__":
    try:
        processor = DataProcessor()
        trainer = ModelTrainer(processor.base_path)
        
        if trainer._check_cached_models():
            print("Found cached models! Loading previous training results...")
            models, evaluation_results, scalers, feature_names = trainer._load_cached_models()
            
            for target, metrics in evaluation_results.items():
                trainer._print_metrics(target, metrics)
        else:
            print("No cached models found. Processing data and training new models...")
            combined_data = processor.load_and_combine_data()
            splits, scalers, feature_names = trainer.prepare_training_data(combined_data)
            models, evaluation_results, scalers, feature_names = trainer.train_and_evaluate_models(
                splits, scalers, feature_names
            )
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
