from rest_framework import serializers
from .models import SimulationResult

class CompanyDataSerializer(serializers.Serializer):
    name = serializers.CharField(default='Company')
    industry = serializers.CharField(default='Technology')
    revenues = serializers.FloatField(required=True)
    profits = serializers.FloatField(required=True)
    market_value = serializers.FloatField(required=True)
    employees = serializers.IntegerField(required=True)
    revenue_growth = serializers.FloatField(required=True)
    profit_margin = serializers.FloatField(required=True)
    costs = serializers.FloatField(required=True)

class DecisionSerializer(serializers.Serializer):
    hiring_rate = serializers.FloatField(default=0.1)
    rd_investment = serializers.FloatField(default=0.15)
    market_expansion = serializers.FloatField(default=0.2)

    def validate(self, data):
        """
        Check that all rates are between 0 and 1.
        """
        for field, value in data.items():
            if not 0 <= value <= 1:
                raise serializers.ValidationError(f"{field} must be between 0 and 1")
        return data

class SimulationInputSerializer(serializers.Serializer):
    company_data = CompanyDataSerializer()
    decisions = DecisionSerializer()
    num_years = serializers.IntegerField(default=5, min_value=1, max_value=20)
    market_scenario = serializers.ChoiceField(
        choices=['baseline', 'optimistic', 'pessimistic'],
        default='baseline'
    )

# In your serializers.py
class ParallelSimulationInputSerializer(serializers.Serializer):
    company_data = serializers.DictField()  # Remove child=FloatField()
    base_decisions = serializers.DictField(child=serializers.FloatField())
    decision_variations = serializers.ListField(
        child=serializers.DictField(child=serializers.FloatField())
    )
    num_years = serializers.IntegerField()
    monte_carlo_sims = serializers.IntegerField()
class SimulationResultSerializer(serializers.ModelSerializer):
    visualizations = serializers.JSONField(default=dict)  # Ensure visualizations are included
    insights = serializers.JSONField(default=dict)  # Ensure insights are included

    class Meta:
        model = SimulationResult
        fields = ['id', 'input_data', 'results', 'insights', 'visualizations', 'is_parallel', 'created_at']
        read_only_fields = fields
