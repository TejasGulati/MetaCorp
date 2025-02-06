from django.urls import path
from .views import (
    SimulationAPIView,
    ParallelSimulationAPIView,
    SimulationResultAPIView
)

app_name = 'ml_pipeline'

urlpatterns = [
    # Endpoint for running single simulation
    path(
        'simulate/',
        SimulationAPIView.as_view(),
        name='simulate'
    ),
    
    # Endpoint for running parallel simulations
    path(
        'simulate/parallel/',
        ParallelSimulationAPIView.as_view(),
        name='simulate-parallel'
    ),
    
    # Endpoint for retrieving simulation results
    path(
        'results/<int:simulation_id>/',
        SimulationResultAPIView.as_view(),
        name='simulation-results'
    ),
]