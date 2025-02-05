from rest_framework import serializers
from core.models import Company, Scenario, Simulation

class CompanySerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = ['id', 'name', 'industry', 'created_at']

class ScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scenario
        fields = ['id', 'company', 'name', 'description', 'parameters', 'created_at']

class SimulationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Simulation
        fields = ['id', 'scenario', 'status', 'results', 'created_at', 'completed_at']