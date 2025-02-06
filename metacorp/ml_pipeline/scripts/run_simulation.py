
# run_simulation.py
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import MetaCorpSimulator

def run_parallel_reality_simulation():
    """Run enhanced parallel reality simulation"""
    try:
        print("Initializing Enhanced MetaCorp Simulator...")
        current_dir = Path(__file__).resolve().parent
        base_path = current_dir.parent
        simulator = MetaCorpSimulator(base_path=base_path)
        
        # Initial company state
        company_data = {
            'name': 'TechCorp',
            'industry': 'Technology',
            'revenues': 1000,  # millions USD
            'profits': 150,    # millions USD
            'market_value': 2000,  # millions USD
            'employees': 5000,
            'revenue_growth': 15,  # percentage
            'profit_margin': 15,   # percentage
            'costs': 850,     # millions USD
        }
        
        # Define strategic variations
        strategies = {
            'base': {
                'hiring_rate': 0.1,        # 10% annual hiring increase
                'rd_investment': 0.15,     # 15% of revenue to R&D
                'market_expansion': 0.2,   # 20% focus on market expansion
            },
            'aggressive': {
                'hiring_rate': 0.2,        # 20% annual hiring increase
                'rd_investment': 0.25,     # 25% of revenue to R&D
                'market_expansion': 0.3,   # 30% focus on market expansion
            },
            'conservative': {
                'hiring_rate': 0.05,       # 5% annual hiring increase
                'rd_investment': 0.1,      # 10% of revenue to R&D
                'market_expansion': 0.1,   # 10% focus on market expansion
            },
            'balanced': {
                'hiring_rate': 0.15,       # 15% annual hiring increase
                'rd_investment': 0.18,     # 18% of revenue to R&D
                'market_expansion': 0.15,  # 15% focus on market expansion
            }
        }
        
        # Run simulation with Monte Carlo iterations
        print("\nSimulating Parallel Business Realities...")
        parallel_results, parallel_insights = simulator.simulate_parallel_realities(
            company_data,
            strategies['base'],
            [strategies['aggressive'], strategies['conservative'], strategies['balanced']],
            num_years=5,
            monte_carlo_sims=50
        )
        
        # Create output directory
        output_dir = base_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create enhanced visualizations
        create_visualizations(parallel_results, output_dir)
        
        # Generate comprehensive report
        generate_report(parallel_results, parallel_insights, company_data, strategies, output_dir)
        
        print("\nSimulation completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        return parallel_results, parallel_insights
        
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        raise

# In the create_visualizations function, replace the current style setup with this enhanced version:

def create_visualizations(results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create enhanced visualizations of simulation results"""
    # Use a built-in style that's guaranteed to work
    plt.style.use('default')
    
    # Set common plot parameters
    plt.rcParams.update({
        'figure.figsize': (15, 10),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'lines.linewidth': 2,
    })
    
    # Market Value Trajectory Plot
    fig, ax = plt.subplots()
    
    # Create color palette
    n_colors = len([k for k in results.keys() if '_mc' not in k])
    colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
    
    color_idx = 0
    for reality, df in results.items():
        if '_mc' not in reality:
            color = colors[color_idx]
            
            # Clean up reality name for legend
            clean_name = reality.replace('_', ' ').title()
            
            # Plot main line
            ax.plot(df['year'], df['market_value'], 
                   label=f'{clean_name} (${df.iloc[-1]["market_value"]:.0f}M)',
                   color=color)
            
            # Add uncertainty bands if available
            if 'market_value_lower' in df.columns:
                ax.fill_between(df['year'],
                              df['market_value_lower'],
                              df['market_value_upper'],
                              color=color, alpha=0.2)
            
            color_idx += 1
    
    ax.set_title('Market Value Trajectories with Uncertainty Bands')
    ax.set_xlabel('Year')
    ax.set_ylabel('Market Value (Millions USD)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'market_value_trajectories.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create additional plots for key metrics
    metrics = ['revenue_growth', 'profit_margin']
    for metric in metrics:
        fig, ax = plt.subplots()
        
        color_idx = 0
        for reality, df in results.items():
            if '_mc' not in reality:
                color = colors[color_idx]
                clean_name = reality.replace('_', ' ').title()
                
                # Plot main line
                ax.plot(df['year'], df[metric],
                       label=f'{clean_name} ({df.iloc[-1][metric]:.1f}%)',
                       color=color)
                
                # Add uncertainty bands if available
                if f'{metric}_lower' in df.columns:
                    ax.fill_between(df['year'],
                                  df[f'{metric}_lower'],
                                  df[f'{metric}_upper'],
                                  color=color, alpha=0.2)
                
                color_idx += 1
        
        metric_title = metric.replace('_', ' ').title()
        ax.set_title(f'{metric_title} Trajectories')
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / f'{metric}_trajectories.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    # Create a correlation heatmap
    plt.figure(figsize=(12, 8))
    
    # Get the final year metrics for each reality
    final_metrics = {}
    for reality, df in results.items():
        if '_mc' not in reality:
            final_metrics[reality] = {
                'Market Value': df.iloc[-1]['market_value'],
                'Revenue Growth': df.iloc[-1]['revenue_growth'],
                'Profit Margin': df.iloc[-1]['profit_margin'],
                'Employees': df.iloc[-1]['employees']
            }
    
    # Convert to DataFrame for correlation
    final_metrics_df = pd.DataFrame(final_metrics).T
    
    # Create correlation heatmap
    sns.heatmap(final_metrics_df.corr(), annot=True, cmap='RdYlBu', center=0)
    plt.title('Correlation of Final Year Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_correlations.png', bbox_inches='tight', dpi=300)
    plt.close()

def generate_report(results: Dict[str, pd.DataFrame],
                   insights: Dict[str, List[str]],
                   initial_state: Dict,
                   strategies: Dict,
                   output_dir: Path):
    """Generate comprehensive simulation report"""
    report_path = output_dir / 'simulation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("MetaCorp Enhanced Parallel Reality Simulation Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Initial State
        f.write("Initial Company State:\n")
        f.write("-" * 20 + "\n")
        for key, value in initial_state.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Strategic Variations
        f.write("Strategic Variations:\n")
        f.write("-" * 20 + "\n")
        for strategy_name, strategy in strategies.items():
            f.write(f"\n{strategy_name.title()} Strategy:\n")
            for key, value in strategy.items():
                f.write(f"  {key}: {value*100:.1f}%\n")
        f.write("\n")
        
        # Simulation Results
        f.write("Simulation Results:\n")
        f.write("-" * 20 + "\n")
        for reality, df in results.items():
            if '_mc' not in reality:
                f.write(f"\n{reality}:\n")
                final_state = df.iloc[-1]
                
                # Calculate average annual changes
                avg_mv_growth = (final_state['market_value'] / df.iloc[0]['market_value']) ** (1/5) - 1
                avg_revenue_growth = df['revenue_growth'].mean()
                avg_profit_margin = df['profit_margin'].mean()
                
                f.write(f"  Final Market Value: ${final_state['market_value']:.0f}M\n")
                f.write(f"  Average Annual Market Value Growth: {avg_mv_growth*100:.1f}%\n")
                f.write(f"  Average Revenue Growth: {avg_revenue_growth:.1f}%\n")
                f.write(f"  Average Profit Margin: {avg_profit_margin:.1f}%\n")
                f.write(f"  Final Employee Count: {int(final_state['employees'])}\n")
                
                if 'market_value_lower' in df.columns:
                    f.write("  Uncertainty Ranges (90% confidence):\n")
                    f.write(f"    Market Value: ${df['market_value_lower'].iloc[-1]:.0f}M - "
                           f"${df['market_value_upper'].iloc[-1]:.0f}M\n")
                    f.write(f"    Revenue Growth: {df['revenue_growth_lower'].iloc[-1]:.1f}% - "
                           f"{df['revenue_growth_upper'].iloc[-1]:.1f}%\n")
                    f.write(f"    Profit Margin: {df['profit_margin_lower'].iloc[-1]:.1f}% - "
                           f"{df['profit_margin_upper'].iloc[-1]:.1f}%\n")
        
        # Key Insights
        f.write("\nKey Insights:\n")
        f.write("-" * 20 + "\n")
        for reality, reality_insights in insights.items():
            if '_mc' not in reality:
                f.write(f"\n{reality}:\n")
                for insight in reality_insights:
                    f.write(f"  - {insight}\n")

if __name__ == "__main__":
    run_parallel_reality_simulation()