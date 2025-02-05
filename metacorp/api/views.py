from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from core.models import Company, Scenario, Simulation
from .serializers import CompanySerializer, ScenarioSerializer, SimulationSerializer
from ml_pipeline.simulation import run_simulation_task

class CompanyViewSet(viewsets.ModelViewSet):
    queryset = Company.objects.all()
    serializer_class = CompanySerializer

class ScenarioViewSet(viewsets.ModelViewSet):
    queryset = Scenario.objects.all()
    serializer_class = ScenarioSerializer
    
    @action(detail=True, methods=['post'])
    def run_simulation(self, request, pk=None):
        scenario = self.get_object()
        simulation = Simulation.objects.create(
            scenario=scenario,
            status='pending'
        )
        
        # Trigger async simulation
        run_simulation_task.delay(simulation.id)
        
        return Response({
            'simulation_id': simulation.id,
            'status': 'pending'
        })

class SimulationViewSet(viewsets.ModelViewSet):
    queryset = Simulation.objects.all()
    serializer_class = SimulationSerializer