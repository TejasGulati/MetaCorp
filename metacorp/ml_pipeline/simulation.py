from celery import shared_task
import numpy as np
from django.utils import timezone
from core.models import Simulation

class SimulationPipeline:
    def __init__(self):
        self.features = ['market_size', 'competition_level', 'initial_investment']
    
    def preprocess_data(self, parameters):
        # Convert parameters to feature vector
        # This is a simple example - expand based on your needs
        processed_data = {
            'market_size': float(parameters.get('market_size', 0)),
            'competition_level': self._encode_competition(parameters.get('competition_level', 'medium')),
            'initial_investment': float(parameters.get('initial_investment', 0))
        }
        return processed_data
    
    def _encode_competition(self, level):
        # Simple encoding of competition level
        levels = {'low': 0, 'medium': 1, 'high': 2}
        return levels.get(level.lower(), 1)
    
    def run_simulation(self, processed_data):
        # Simplified simulation logic
        results = {
            'projected_revenue': processed_data['market_size'] * 0.1,
            'risk_level': self._calculate_risk(processed_data),
            'success_probability': self._calculate_success_prob(processed_data)
        }
        return results
    
    def _calculate_risk(self, data):
        # Simple risk calculation
        if data['competition_level'] == 2:  # High competition
            return 'high'
        elif data['market_size'] < 100000:
            return 'medium'
        return 'low'
    
    def _calculate_success_prob(self, data):
        # Simple success probability calculation
        base_prob = 0.7
        if data['competition_level'] == 2:
            base_prob -= 0.2
        if data['market_size'] > 1000000:
            base_prob += 0.1
        return min(max(base_prob, 0), 1)  # Ensure between 0 and 1

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
        if simulation:
            simulation.status = 'failed'
            simulation.save()
        raise e