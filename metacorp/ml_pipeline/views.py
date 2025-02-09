from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.conf.urls.static import static
from pathlib import Path
from .serializers import (
    SimulationInputSerializer,
    ParallelSimulationInputSerializer,
    SimulationResultSerializer
)
from .models import SimulationResult
from ml_pipeline.scripts.simulation import MetaCorpSimulator
import json
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import uuid
import os

def save_plot_to_static(fig, plot_name):
    """Save plot to static directory and return its URL path"""
    # Create a unique filename using UUID
    filename = f"{plot_name}_{uuid.uuid4()}.png"

    # Define the path relative to STATICFILES_DIRS
    plot_dir = os.path.join(settings.STATICFILES_DIRS[0], 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Full path to save the file
    file_path = os.path.join(plot_dir, filename)

    # Save the plot
    fig.savefig(file_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    # Return the URL path
    return f"{settings.STATIC_URL}plots/{filename}"


def create_visualizations(results, output_dir):
    """Create visualizations and return their URLs"""
    try:
        image_urls = {}
        
        # Set common plot parameters
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (15, 10),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'lines.linewidth': 2,
        })
        
        # Ensure proper data format
        formatted_results = {}
        for reality, data in results.items():
            if isinstance(data, list):
                formatted_results[reality] = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                formatted_results[reality] = data
            else:
                formatted_results[reality] = pd.DataFrame(data)
        
        # Market Value Trajectory Plot
        plt.figure(figsize=(15, 10))
        n_colors = len([k for k in formatted_results.keys() if '_mc' not in k])
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_colors, 1)))
        
        color_idx = 0
        for reality, df in formatted_results.items():
            if '_mc' not in reality:
                color = colors[color_idx]
                clean_name = reality.replace('_', ' ').title()
                
                plt.plot(df['year'], df['market_value'], 
                        label=f'{clean_name} (${df.iloc[-1]["market_value"]:.0f}M)',
                        color=color)
                color_idx += 1
        
        plt.title('Market Value Trajectories')
        plt.xlabel('Year')
        plt.ylabel('Market Value (Millions USD)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save market value plot and get URL
        image_urls['market_value_trajectories'] = save_plot_to_static(plt.gcf(), 'market_value')
        plt.close()
        
        # Create additional plots for key metrics
        metrics = ['revenue_growth', 'profit_margin']
        for metric in metrics:
            plt.figure(figsize=(15, 10))
            color_idx = 0
            for reality, df in formatted_results.items():
                if '_mc' not in reality:
                    color = colors[color_idx]
                    clean_name = reality.replace('_', ' ').title()
                    
                    plt.plot(df['year'], df[metric],
                            label=f'{clean_name} ({df.iloc[-1][metric]:.1f}%)',
                            color=color)
                    color_idx += 1
            
            metric_title = metric.replace('_', ' ').title()
            plt.title(f'{metric_title} Trajectories')
            plt.xlabel('Year')
            plt.ylabel('Percentage (%)')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save metric plot and get URL
            image_urls[f'{metric}_trajectories'] = save_plot_to_static(plt.gcf(), metric)
            plt.close()
        
        # Create metrics correlation heatmap
        plt.figure(figsize=(12, 8))
        final_metrics = {}
        for reality, df in formatted_results.items():
            if '_mc' not in reality:
                final_metrics[reality] = {
                    'Market Value': df.iloc[-1]['market_value'],
                    'Revenue Growth': df.iloc[-1]['revenue_growth'],
                    'Profit Margin': df.iloc[-1]['profit_margin'],
                    'Employees': df.iloc[-1]['employees']
                }
        
        if final_metrics:
            final_metrics_df = pd.DataFrame(final_metrics).T
            sns.heatmap(final_metrics_df.corr(), annot=True, cmap='RdYlBu', center=0, 
                       vmin=-1, vmax=1)
            plt.title('Correlation of Final Year Metrics')
            plt.tight_layout()
            
            # Save heatmap and get URL
            image_urls['metric_correlations'] = save_plot_to_static(plt.gcf(), 'correlations')
            plt.close()
        
        return image_urls
    
    except Exception as e:
        print(f"Error in create_visualizations: {str(e)}")
        return {}

class SimulationAPIView(APIView):
    """API endpoint for running single business simulations"""
    def post(self, request):
        serializer = SimulationInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Initialize simulator
            base_path = Path(settings.BASE_DIR)
            simulator = MetaCorpSimulator(base_path=base_path)
            
            # Extract validated data
            data = serializer.validated_data
            company_data = {
                'name': data['company_data'].get('name', 'Company'),
                'industry': data['company_data'].get('industry', 'Technology'),
                'revenues': float(data['company_data'].get('revenues', 0)),
                'profits': float(data['company_data'].get('profits', 0)),
                'market_value': float(data['company_data'].get('market_value', 0)),
                'employees': int(data['company_data'].get('employees', 0)),
                'revenue_growth': float(data['company_data'].get('revenue_growth', 0)),
                'profit_margin': float(data['company_data'].get('profit_margin', 0)),
                'costs': float(data['company_data'].get('costs', 0))
            }
            
            decisions = {
                'hiring_rate': float(data['decisions'].get('hiring_rate', 0.1)),
                'rd_investment': float(data['decisions'].get('rd_investment', 0.15)),
                'market_expansion': float(data['decisions'].get('market_expansion', 0.2))
            }
            
            num_years = int(data.get('num_years', 5))
            market_scenario = data.get('market_scenario', 'baseline')
            
            # Run simulation
            results, insights = simulator.simulate_scenario(
                company_data=company_data,
                decisions=decisions,
                num_years=num_years,
                market_scenario=market_scenario
            )
            
            # Create visualizations
            output_dir = base_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert results to dictionary format for visualization
            results_dict = {'baseline': results}
            visualizations = create_visualizations(results_dict, output_dir)
            
            # Save results to database
            simulation_result = SimulationResult.objects.create(
                input_data=serializer.validated_data,
                results=results.to_dict(orient='records'),
                insights=insights,
                visualizations=visualizations
            )
            
            # Prepare response data
            response_data = {
                'id': simulation_result.id,
                'input_data': simulation_result.input_data,
                'results': simulation_result.results,
                'insights': simulation_result.insights,
                'visualizations': visualizations,
                'is_parallel': simulation_result.is_parallel,
                'created_at': simulation_result.created_at
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ParallelSimulationAPIView(APIView):
    """API endpoint for running parallel business simulations"""
    def post(self, request):
        serializer = ParallelSimulationInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Initialize simulator
            base_path = Path(settings.BASE_DIR)
            simulator = MetaCorpSimulator(base_path=base_path)
            
            # Extract validated data
            data = serializer.validated_data
            company_data = {
                'name': data['company_data'].get('name', 'Company'),
                'industry': data['company_data'].get('industry', 'Technology'),
                'revenues': float(data['company_data'].get('revenues', 0)),
                'profits': float(data['company_data'].get('profits', 0)),
                'market_value': float(data['company_data'].get('market_value', 0)),
                'employees': int(data['company_data'].get('employees', 0)),
                'revenue_growth': float(data['company_data'].get('revenue_growth', 0)),
                'profit_margin': float(data['company_data'].get('profit_margin', 0)),
                'costs': float(data['company_data'].get('costs', 0))
            }
            
            base_decisions = {
                'hiring_rate': float(data['base_decisions'].get('hiring_rate', 0.1)),
                'rd_investment': float(data['base_decisions'].get('rd_investment', 0.15)),
                'market_expansion': float(data['base_decisions'].get('market_expansion', 0.2))
            }
            
            decision_variations = []
            for variation in data['decision_variations']:
                decision_variations.append({
                    'hiring_rate': float(variation.get('hiring_rate', 0.1)),
                    'rd_investment': float(variation.get('rd_investment', 0.15)),
                    'market_expansion': float(variation.get('market_expansion', 0.2))
                })
            
            num_years = int(data.get('num_years', 5))
            monte_carlo_sims = int(data.get('monte_carlo_sims', 50))
            
            # Run parallel simulation
            parallel_results, parallel_insights = simulator.simulate_parallel_realities(
                company_data=company_data,
                base_decisions=base_decisions,
                decision_variations=decision_variations,
                num_years=num_years,
                monte_carlo_sims=monte_carlo_sims
            )
            
            # Create visualizations
            output_dir = base_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            visualizations = create_visualizations(parallel_results, output_dir)
            
            # Convert results to serializable format
            formatted_results = {
                k: v.to_dict(orient='records') 
                for k, v in parallel_results.items()
            }
            
            # Save results to database
            simulation_result = SimulationResult.objects.create(
                input_data=serializer.validated_data,
                results=formatted_results,
                insights=parallel_insights,
                visualizations=visualizations,
                is_parallel=True
            )
            
            # Prepare response data
            response_data = {
                'id': simulation_result.id,
                'input_data': simulation_result.input_data,
                'results': simulation_result.results,
                'insights': simulation_result.insights,
                'visualizations': visualizations,
                'is_parallel': simulation_result.is_parallel,
                'created_at': simulation_result.created_at
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            print(f"Error in parallel simulation: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
class SimulationResultAPIView(APIView):
    """API endpoint for retrieving simulation results"""
    def get(self, request, simulation_id):
        try:
            result = SimulationResult.objects.get(id=simulation_id)
            
            # Reconstruct results DataFrame if needed
            if result.is_parallel:
                formatted_results = {
                    k: pd.DataFrame(v) for k, v in result.results.items()
                }
            else:
                formatted_results = {'baseline': pd.DataFrame(result.results)}
            
            # Create fresh visualizations for the stored results
            output_dir = Path(settings.BASE_DIR) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            visualizations = create_visualizations(formatted_results, output_dir)
            
            # Update the result object with new visualization URLs
            result.visualizations = visualizations
            result.save()
            
            # Prepare response with updated visualizations
            response_data = {
                'id': result.id,
                'input_data': result.input_data,
                'results': result.results,
                'insights': result.insights,
                'visualizations': visualizations,  # Include the fresh visualizations
                'is_parallel': result.is_parallel,
                'created_at': result.created_at
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except SimulationResult.DoesNotExist:
            return Response(
                {'error': 'Simulation result not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': f'Error retrieving simulation result: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )