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
        
        # Enhanced model parameters with focus on market value prediction
        self.model_params = {
            'market_value': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators':2000,  # Increased from 500
                    'max_depth': 8,        # Reduced from 8 to prevent overfitting
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'loss': 'huber',
                    'random_state': 42,
                    'validation_fraction': 0.1,
                    'n_iter_no_change': 30,
                    'tol': 1e-4
                }
            },
            # Other models' parameters remain unchanged
            'profit_margin': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 400,
                    'max_depth': 6,
                    'min_samples_split': 5,
                    'min_samples_leaf': 3,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'loss': 'huber',
                    'random_state': 42
                }
            },
            'revenue_growth': {
                'model_type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 400,
                    'max_depth': 5,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'loss': 'huber',
                    'random_state': 42
                }
            }
        }

    def _engineer_features(self, df):
        """Enhanced feature engineering with focus on market value prediction"""
        df = df.copy()
        
        # Basic financial ratios
        df['profit_growth'] = df['profits'] / df['revenues'] * 100
        df['profit_per_employee'] = df['profits'] / df['employees']
        df['revenue_per_employee'] = df['revenues'] / df['employees']
        
        # New advanced financial metrics
        df['profit_to_employee_ratio'] = np.where(
            df['employees'] > 0,
            df['profits'] / df['employees'],
            0
        )
        
        # Industry-specific metrics
        industry_avg_market_value = df.groupby('industry')['market_value'].transform('mean')
        industry_avg_revenue = df.groupby('industry')['revenues'].transform('mean')
        df['industry_market_value_ratio'] = df['market_value'] / industry_avg_market_value
        df['industry_revenue_ratio'] = df['revenues'] / industry_avg_revenue
        
        # Size and efficiency metrics
        df['size_score'] = (
            (df['revenues'] / df['revenues'].max()) * 0.4 +
            (df['employees'] / df['employees'].max()) * 0.3 +
            (df['market_value'] / df['market_value'].max()) * 0.3
        )
        
        df['efficiency_score'] = (
            (df['profit_margin'] / df['profit_margin'].max()) * 0.4 +
            (df['revenue_growth'] / df['revenue_growth'].max()) * 0.3 +
            (df['employee_productivity'] / df['employee_productivity'].max()) * 0.3
        )
        
        # Interaction terms
        df['revenue_profit_interaction'] = df['revenues'] * df['profits']
        df['market_efficiency_interaction'] = df['market_value'] * df['efficiency_score']
        
        return df

    def prepare_training_data(self, df):
        try:
            df = self._engineer_features(df)
            
            # Extended feature set for market value prediction
            numerical_features = [
                'revenues', 'profits', 'employees', 'revenue_growth',
                'profit_margin', 'employee_productivity',
                'profit_per_employee', 'revenue_per_employee', 'profit_growth',
                'efficiency_score', 'size_score', 'profit_to_employee_ratio',
                'industry_market_value_ratio', 'industry_revenue_ratio',
                'revenue_profit_interaction'
            ]
            
            # Remove market_to_revenue_ratio to prevent data leakage
            if 'market_to_revenue_ratio' in numerical_features:
                numerical_features.remove('market_to_revenue_ratio')
                
            categorical_features = ['industry']
            target_variables = ['market_value', 'profit_margin', 'revenue_growth']
            
            # Enhanced outlier handling
            df = self._handle_outliers(df, numerical_features)
            
            X = df[numerical_features + categorical_features].copy()
            
            # Improved feature scaling
            for feature in numerical_features:
                if feature not in ['revenue_growth', 'profit_margin']:
                    min_val = X[feature].min()
                    offset = abs(min_val) + 1 if min_val < 0 else 0
                    X[feature] = np.log1p(X[feature] + offset)
            
            # Enhanced industry encoding
            X = pd.get_dummies(X, columns=['industry'], drop_first=True)
            
            # Target variable transformations
            y = df[target_variables].copy()
            y['market_value'] = np.log1p(y['market_value'])
            
            joblib.dump(list(X.columns), self.models_path / 'feature_names.joblib')
            
            # Stratified split with more sophisticated binning
            splits = {}
            for target in target_variables:
                try:
                    bins = pd.qcut(y[target], q=10, labels=False, duplicates='drop')
                except ValueError:
                    bins = pd.qcut(y[target], q=5, labels=False, duplicates='drop')
                
                splits[target] = train_test_split(
                    X, y[target],
                    test_size=0.2,
                    random_state=42,
                    stratify=bins
                )
            
            return splits
        
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            raise

    def _handle_outliers(self, df, numerical_features):
        """Enhanced outlier handling with feature-specific approaches"""
        df = df.copy()
        
        for feature in numerical_features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            # Feature-specific thresholds
            if feature == 'market_value':
                threshold = 7.0  # Most lenient for market value
            elif feature in ['revenues', 'profits']:
                threshold = 5.0  # Lenient for main financial metrics
            elif feature in ['revenue_growth', 'profit_margin']:
                threshold = 3.0  # Moderate for percentage metrics
            else:
                threshold = 2.0  # Stricter for derived metrics
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Log-based outlier handling for highly skewed features
            if feature in ['market_value', 'revenues', 'profits']:
                df[feature] = np.where(
                    df[feature] > upper_bound,
                    upper_bound * (1 + np.log1p(df[feature] / upper_bound)),
                    df[feature]
                )
            else:
                df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def train_and_evaluate_models(self, splits):
        evaluation_results = {}
        models = {}
        
        for target, (X_train, X_test, y_train, y_test) in splits.items():
            try:
                print(f"\nTraining and evaluating model for: {target}")
                
                # Get model configuration
                model_config = self.model_params[target]
                model = model_config['model_type'](**model_config['params'])
                
                # Perform k-fold cross-validation with shuffling
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                
                # Fit model
                model.fit(X_train, y_train)
                models[target] = model
                
                # Make predictions and reverse transformations if necessary
                y_pred = model.predict(X_test)
                y_pred_train = model.predict(X_train)
                
                if target == 'market_value':
                    y_test_orig = np.expm1(y_test)
                    y_pred_orig = np.expm1(y_pred)
                    y_train_orig = np.expm1(y_train)
                    y_pred_train_orig = np.expm1(y_pred_train)
                else:
                    y_test_orig = y_test
                    y_pred_orig = y_pred
                    y_train_orig = y_train
                    y_pred_train_orig = y_pred_train
                
                # Calculate metrics
                metrics = {
                    'R2_Score_Test': r2_score(y_test_orig, y_pred_orig),
                    'R2_Score_Train': r2_score(y_train_orig, y_pred_train_orig),
                    'MAE': mean_absolute_error(y_test_orig, y_pred_orig),
                    'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
                    'CV_R2_Mean': cv_scores.mean(),
                    'CV_R2_Std': cv_scores.std(),
                    'Feature_Importance': dict(zip(X_train.columns, model.feature_importances_))
                }
                
                evaluation_results[target] = metrics
                
                # Save model
                joblib.dump(model, self.models_path / f'{target}_model.joblib')
                
                # Print detailed metrics
                self._print_metrics(target, metrics)
                
            except Exception as e:
                print(f"Error training model for {target}: {str(e)}")
                continue
        
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
        # Initialize processor and load data
        processor = DataProcessor()
        combined_data = processor.load_and_combine_data()
        
        # Initialize trainer
        trainer = ModelTrainer(processor.base_path)
        
        # Prepare data with train-test split
        splits = trainer.prepare_training_data(combined_data)
        
        # Train and evaluate models
        models, evaluation_results = trainer.train_and_evaluate_models(splits)
        
        print("\nModel training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise