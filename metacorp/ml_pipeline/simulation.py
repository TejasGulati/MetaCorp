from celery import shared_task
import numpy as np
from django.utils import timezone
from core.models import Simulation
import joblib
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SimulationPipeline:
    def __init__(self):
        self.models_path = Path(__file__).parent / "data/models"
        
        # Load trained models and preprocessors
        try:
            self.models = {
                'market_value': joblib.load(self.models_path / 'market_value_model.joblib'),
                'profit_margin': joblib.load(self.models_path / 'profit_margin_model.joblib'),
                'revenue_growth': joblib.load(self.models_path / 'revenue_growth_model.joblib')
            }
            self.scaler = joblib.load(self.models_path / 'scaler.joblib')
            self.label_encoder = joblib.load(self.models_path / 'label_encoder.joblib')
            self.ml_enabled = True
        except FileNotFoundError:
            self.ml_enabled = False
            # Fallback features if models aren't trained yet
            self.features = ['market_size', 'competition_level', 'initial_investment']

    def preprocess_data(self, parameters):
        if not self.ml_enabled:
            return self._legacy_preprocess(parameters)

        try:
            # Convert parameters to feature vector for ML models
            features = {
                'revenues': float(parameters.get('revenues', 0)),
                'employees': float(parameters.get('employees', 0)),
                'revenue_growth': float(parameters.get('revenue_growth', 0)),
                'profit_margin': float(parameters.get('profit_margin', 0)),
                'employee_productivity': float(parameters.get('revenues', 0)) / float(parameters.get('employees', 1)),
                'industry': parameters.get('industry', 'Technology')
            }

            # Transform industry using label encoder
            industry_encoded = self.label_encoder.transform([features['industry']])[0]

            # Scale numerical features
            numerical_features = ['revenues', 'employees', 'revenue_growth', 'profit_margin', 'employee_productivity']
            feature_values = [[features[f] for f in numerical_features]]
            scaled_features = self.scaler.transform(feature_values)

            # Combine numerical and categorical features
            processed_data = np.concatenate([scaled_features, [[industry_encoded]]], axis=1)
            return processed_data

        except Exception as e:
            # Fallback to legacy processing if any issues
            return self._legacy_preprocess(parameters)

    def _legacy_preprocess(self, parameters):
        # Original preprocessing logic as fallback
        return {
            'market_size': float(parameters.get('market_size', 0)),
            'competition_level': self._encode_competition(parameters.get('competition_level', 'medium')),
            'initial_investment': float(parameters.get('initial_investment', 0))
        }

    def _encode_competition(self, level):
        levels = {'low': 0, 'medium': 1, 'high': 2}
        return levels.get(level.lower(), 1)

    def run_simulation(self, processed_data):
        if not self.ml_enabled:
            return self._legacy_simulation(processed_data)

        try:
            # Get predictions from ML models
            predictions = {
                'market_value': float(self.models['market_value'].predict(processed_data)[0]),
                'profit_margin': float(self.models['profit_margin'].predict(processed_data)[0]),
                'revenue_growth': float(self.models['revenue_growth'].predict(processed_data)[0])
            }

            # Calculate derived metrics
            projected_revenue = predictions['revenue_growth'] * processed_data[0][0]  # Using scaled revenues
            risk_level = self._calculate_risk_ml(predictions)
            success_prob = self._calculate_success_prob_ml(predictions)

            results = {
                'projected_revenue': projected_revenue,
                'market_value_prediction': predictions['market_value'],
                'profit_margin_prediction': predictions['profit_margin'],
                'revenue_growth_prediction': predictions['revenue_growth'],
                'risk_level': risk_level,
                'success_probability': success_prob
            }
            return results

        except Exception as e:
            # Fallback to legacy simulation if any issues
            return self._legacy_simulation(processed_data)

    def _legacy_simulation(self, processed_data):
        return {
            'projected_revenue': processed_data['market_size'] * 0.1,
            'risk_level': self._calculate_risk(processed_data),
            'success_probability': self._calculate_success_prob(processed_data)
        }

    def _calculate_risk_ml(self, predictions):
        risk_score = 0
        
        # Risk based on growth
        if predictions['revenue_growth'] < 0:
            risk_score += 2
        elif predictions['revenue_growth'] < 5:
            risk_score += 1
        
        # Risk based on profit margin
        if predictions['profit_margin'] < 10:
            risk_score += 2
        elif predictions['profit_margin'] < 20:
            risk_score += 1

        return 'high' if risk_score >= 3 else 'medium' if risk_score >= 1 else 'low'

    def _calculate_success_prob_ml(self, predictions):
        base_prob = 0.7
        
        # Adjust based on growth and margin predictions
        base_prob += min(0.2, predictions['revenue_growth'] / 100)
        base_prob += min(0.1, predictions['profit_margin'] / 200)
        
        return min(max(base_prob, 0), 1)

    def _calculate_risk(self, data):
        # Legacy risk calculation
        if data['competition_level'] == 2:
            return 'high'
        elif data['market_size'] < 100000:
            return 'medium'
        return 'low'

    def _calculate_success_prob(self, data):
        # Legacy success probability calculation
        base_prob = 0.7
        if data['competition_level'] == 2:
            base_prob -= 0.2
        if data['market_size'] > 1000000:
            base_prob += 0.1
        return min(max(base_prob, 0), 1)

@shared_task
def run_simulation_task(simulation_id):
    try:
        simulation = Simulation.objects.get(id=simulation_id)
        simulation.status = 'running'
        simulation.save()

        # Run simulation
        pipeline = SimulationPipeline()
        processed_data = pipeline.preprocess_data(simulation.scenario.parameters)
        results = pipeline.run_simulation(processed_data)

        # Update simulation with results
        simulation.status = 'completed'
        simulation.results = results
        simulation.completed_at = timezone.now()
        simulation.save()

        return results

    except Exception as e:
        if 'simulation' in locals():
            simulation.status = 'failed'
            simulation.save()
        raise e