# simulation.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import random
from scipy.stats import norm

class MetaCorpSimulator:
    """
    Advanced business simulator that models parallel business realities to predict
    the impact of different strategic decisions and market conditions.
    """
    
    def __init__(self, base_path: str = None):
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path.cwd()
        
        self.models_path = self.base_path / "data" / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
            
        try:
            self.models = self._load_models()
        except FileNotFoundError as e:
            print(f"Warning: Unable to load models. Using default models. Error: {e}")
            self._initialize_default_models()
        
        # Initialize industry-specific parameters with more realistic constraints
        self._init_industry_parameters()

    def _init_industry_parameters(self):
        """Initialize industry-specific parameters with refined constraints"""
        self.industry_params = {
            'Technology': {
                'max_profit_margin': 0.30,  # Reduced from 0.35
                'min_profit_margin': -0.10,  # Increased from -0.15
                'max_growth_rate': 0.40,    # Reduced from 0.50
                'min_growth_rate': -0.20,   # Added minimum growth rate
                'volatility': 0.20,         # Reduced from 0.25
                'rd_effectiveness': 1.2,
                'max_employee_growth': 0.30, # Added employee growth constraint
                'min_employee_growth': -0.15 # Added employee decline constraint
            },
            'Manufacturing': {
                'max_profit_margin': 0.20,   # Reduced from 0.25
                'min_profit_margin': -0.08,  # Increased from -0.10
                'max_growth_rate': 0.25,     # Reduced from 0.30
                'min_growth_rate': -0.15,
                'volatility': 0.12,          # Reduced from 0.15
                'rd_effectiveness': 0.8,
                'max_employee_growth': 0.25,
                'min_employee_growth': -0.12
            },
            'Services': {
                'max_profit_margin': 0.25,   # Reduced from 0.30
                'min_profit_margin': -0.09,  # Increased from -0.12
                'max_growth_rate': 0.30,     # Reduced from 0.35
                'min_growth_rate': -0.18,
                'volatility': 0.15,          # Reduced from 0.20
                'rd_effectiveness': 1.0,
                'max_employee_growth': 0.28,
                'min_employee_growth': -0.14
            }
        }

    def _load_models(self) -> Dict:
        """Load trained models and scalers for predictions"""
        models = {}
        for target in ['market_value', 'profit_margin', 'revenue_growth']:
            model_path = self.models_path / f'{target}_model.joblib'
            
            try:
                if model_path.exists():
                    models[target] = {
                        'model': joblib.load(model_path),
                        'scaler': None  # Will be initialized during first prediction
                    }
                    print(f"Loaded model for {target}")
                else:
                    print(f"Model file not found for {target}")
                    raise FileNotFoundError(f"Model file not found for {target}")
                    
            except Exception as e:
                print(f"Error loading model for {target}: {e}")
                raise FileNotFoundError(f"Error loading model for {target}")
                
        return models

    def _initialize_default_models(self):
        """Initialize default models for parallel reality simulation"""
        from sklearn.ensemble import GradientBoostingRegressor
        
        self.models = {}
        for target in ['market_value', 'profit_margin', 'revenue_growth']:
            # Create a simple model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Initialize with some dummy data
            X = np.random.rand(100, 5)  # 5 features
            y = np.random.rand(100)
            model.fit(X, y)
            
            self.models[target] = {
                'model': model,
                'scaler': None
            }
            
            # Save the model
            model_path = self.models_path / f'{target}_model.joblib'
            joblib.dump(model.fit(X, y), model_path)
            print(f"Initialized and saved default model for {target}")


    def simulate_parallel_realities(self,
                                  company_data: Dict,
                                  base_decisions: Dict,
                                  decision_variations: List[Dict],
                                  num_years: int = 5,
                                  market_scenarios: Optional[List[str]] = None,
                                  monte_carlo_sims: int = 10) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        """
        Simulate multiple parallel business realities with Monte Carlo variations.
        """
        if market_scenarios is None:
            market_scenarios = ['baseline', 'optimistic', 'pessimistic']
            
        parallel_results = {}
        parallel_insights = {}
        
        industry = company_data.get('industry', 'Technology')
        industry_params = self.industry_params.get(industry, self.industry_params['Technology'])
        
        # Run Monte Carlo simulations for each scenario
        for mc_sim in range(monte_carlo_sims):
            # Simulate base reality with Monte Carlo variation
            base_results, base_insights = self.simulate_scenario(
                company_data,
                base_decisions,
                num_years,
                'baseline',
                industry_params,
                random_seed=mc_sim
            )
            
            if mc_sim == 0:  # Keep only the first base reality
                parallel_results['base_reality'] = base_results
                parallel_insights['base_reality'] = base_insights
            
            # Simulate decision variations
            with ThreadPoolExecutor() as executor:
                futures = []
                for i, decisions in enumerate(decision_variations):
                    for scenario in market_scenarios:
                        reality_key = f"reality_{i+1}_{scenario}_mc{mc_sim}"
                        future = executor.submit(
                            self.simulate_scenario,
                            company_data.copy(),
                            decisions,
                            num_years,
                            scenario,
                            industry_params,
                            random_seed=mc_sim
                        )
                        futures.append((reality_key, future))
                
                # Collect results
                for reality_key, future in futures:
                    results, insights = future.result()
                    if mc_sim == 0:  # Keep only first simulation for each variation
                        clean_key = reality_key.replace('_mc0', '')
                        parallel_results[clean_key] = results
                        parallel_insights[clean_key] = insights
        
        # Aggregate Monte Carlo results for uncertainty bands
        self._add_uncertainty_bands(parallel_results, monte_carlo_sims)
        
        return parallel_results, parallel_insights

    def _add_uncertainty_bands(self, results: Dict[str, pd.DataFrame], num_sims: int):
        """Add uncertainty bands based on Monte Carlo simulations"""
        metrics = ['market_value', 'profit_margin', 'revenue_growth']
        
        for base_key in list(results.keys()):
            if '_mc' not in base_key:
                mc_keys = [f"{base_key}_mc{i}" for i in range(num_sims)]
                mc_keys = [k for k in mc_keys if k in results]
                
                if not mc_keys:
                    continue
                
                for metric in metrics:
                    values = np.array([results[k][metric].values for k in mc_keys])
                    results[base_key][f'{metric}_lower'] = np.percentile(values, 10, axis=0)
                    results[base_key][f'{metric}_upper'] = np.percentile(values, 90, axis=0)

    def simulate_scenario(self,
                         company_data: Dict,
                         decisions: Dict,
                         num_years: int = 5,
                         market_scenario: str = 'baseline',
                         industry_params: Dict = None,
                         random_seed: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Simulate a single business reality with market conditions and industry constraints.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        if industry_params is None:
            industry_params = self.industry_params['Technology']
            
        results = []
        current_state = company_data.copy()
        insights = []
        
        scenario_modifiers = self._get_scenario_modifiers(market_scenario)
        
        # Add market cycle simulation
        market_cycle = self._generate_market_cycle(num_years, scenario_modifiers['volatility'])
        
        for year in range(num_years):
            # Apply market cycle effect
            scenario_modifiers['cycle_effect'] = market_cycle[year]
            
            # Apply strategic decisions with market conditions
            current_state = self._apply_decisions_with_market_conditions(
                current_state,
                decisions,
                scenario_modifiers,
                industry_params
            )
            
            # Predict metrics with industry constraints
            predictions = self._predict_metrics_with_uncertainty(
                current_state,
                scenario_modifiers,
                industry_params
            )
            
            # Apply industry constraints
            predictions = self._apply_industry_constraints(predictions, industry_params)
            
            # Update state
            current_state.update(predictions)
            current_state['year'] = year + 1
            current_state['scenario'] = market_scenario
            
            results.append(current_state.copy())
            
            if year > 0:
                insight = self._analyze_changes(
                    results[-2],
                    results[-1],
                    market_scenario,
                    industry_params
                )
                insights.append(f"Year {year + 1}: {insight}")
        
        return pd.DataFrame(results), insights

    def _get_scenario_modifiers(self, scenario: str) -> Dict[str, float]:
        """Get market condition modifiers with more realistic randomization"""
        base_scenarios = {
            'baseline': {
                'growth_modifier': 1.0,
                'risk_modifier': 1.0,
                'market_sentiment': 1.0,
                'volatility': 0.12  # Reduced from 0.15
            },
            'optimistic': {
                'growth_modifier': 1.15,  # Reduced from 1.2
                'risk_modifier': 0.85,    # Increased from 0.8
                'market_sentiment': 1.2,   # Reduced from 1.3
                'volatility': 0.15        # Reduced from 0.2
            },
            'pessimistic': {
                'growth_modifier': 0.85,   # Increased from 0.8
                'risk_modifier': 1.25,     # Reduced from 1.4
                'market_sentiment': 0.8,   # Increased from 0.7
                'volatility': 0.18        # Reduced from 0.25
            }
        }
        
        # Add smaller random variation to modifiers
        scenario_mods = base_scenarios.get(scenario, base_scenarios['baseline'])
        return {k: v * random.uniform(0.95, 1.05) for k, v in scenario_mods.items()}  # Reduced randomness

    def _generate_market_cycle(self, num_years: int, volatility: float) -> List[float]:
        """Generate realistic market cycle effects with dampened volatility"""
        # Generate smoother market cycle using sine wave
        t = np.linspace(0, 2 * np.pi, num_years)
        base_cycle = np.sin(t) * (volatility * 0.8)  # Reduced amplitude
        
        # Add smaller random walk component
        random_walk = np.cumsum(np.random.normal(0, volatility/3, num_years))  # Reduced from volatility/2
        
        # Combine and normalize with dampened effect
        combined = base_cycle + random_walk
        return (combined - np.mean(combined)) * (volatility * 0.9)

    def _apply_decisions_with_market_conditions(self,
                                              state: Dict,
                                              decisions: Dict,
                                              scenario_modifiers: Dict,
                                              industry_params: Dict) -> Dict:
        """Apply strategic decisions with more realistic market impacts"""
        new_state = state.copy()
        
        # Get modifiers with dampened effects
        growth_mod = scenario_modifiers['growth_modifier']
        risk_mod = scenario_modifiers['risk_modifier']
        cycle_effect = scenario_modifiers.get('cycle_effect', 0) * 0.8  # Dampened cycle effect
        
        # Apply market cycle effect with smoother transition
        market_multiplier = 1 + (cycle_effect * 0.9)  # Reduced multiplier effect
        
        # Workforce changes with stricter constraints
        if 'hiring_rate' in decisions:
            max_growth = min(decisions['hiring_rate'], industry_params['max_employee_growth'])
            min_growth = max(decisions['hiring_rate'], industry_params['min_employee_growth'])
            
            # Calculate actual growth with more realistic factors
            actual_growth = np.clip(
                decisions['hiring_rate'] * growth_mod * market_multiplier,
                min_growth,
                max_growth
            )
            
            # Add more realistic employee churn
            base_churn = 0.08  # Base annual churn rate
            churn_rate = base_churn * risk_mod * random.uniform(0.9, 1.1)
            net_growth = actual_growth - churn_rate
            
            # Apply smoother employee growth
            new_state['employees'] *= (1 + np.clip(net_growth, 
                                                 industry_params['min_employee_growth'],
                                                 industry_params['max_employee_growth']))
        
        # R&D investment impact with more realistic diminishing returns
        if 'rd_investment' in decisions:
            rd_effectiveness = industry_params['rd_effectiveness']
            # Enhanced diminishing returns formula
            diminishing_return = 1 / (1 + np.square(decisions['rd_investment']))
            
            revenue_impact = (decisions['rd_investment'] * 0.08 * growth_mod * 
                            rd_effectiveness * diminishing_return * market_multiplier)
            cost_impact = (decisions['rd_investment'] * 0.06 * risk_mod * 
                         rd_effectiveness * market_multiplier)
            
            # Apply impacts with constraints
            new_state['revenues'] *= (1 + np.clip(revenue_impact, -0.15, 0.25))
            new_state['costs'] *= (1 + np.clip(cost_impact, -0.10, 0.20))
        
        # Market expansion with more realistic success rates
        if 'market_expansion' in decisions:
            success_rate = self._calculate_expansion_success_rate(
                new_state,
                decisions['market_expansion'],
                scenario_modifiers
            )
            
            # More conservative impact calculations
            revenue_impact = (decisions['market_expansion'] * 0.12 * 
                            success_rate * growth_mod * market_multiplier)
            cost_impact = (decisions['market_expansion'] * 0.10 * 
                         risk_mod * market_multiplier)
            
            # Apply impacts with constraints
            new_state['revenues'] *= (1 + np.clip(revenue_impact, -0.12, 0.20))
            new_state['costs'] *= (1 + np.clip(cost_impact, -0.08, 0.15))
        
        return new_state

    def _calculate_expansion_success_rate(self,
                                        state: Dict,
                                        expansion_rate: float,
                                        scenario_modifiers: Dict) -> float:
        """Calculate market expansion success rate with more realistic factors"""
        # More conservative base success rate
        base_success = random.uniform(0.65, 0.85)  # Narrowed range
        
        # Enhanced company size factor with logarithmic scaling
        size_factor = np.log1p(state['revenues'] / 1000) / np.log1p(10000)
        size_factor = np.clip(size_factor, 0.3, 0.9)
        
        # Market conditions factor with dampened effect
        market_factor = scenario_modifiers['market_sentiment'] * 0.8
        
        # Experience factor with more realistic scaling
        experience_factor = min(0.8, state['revenue_growth'] / 40)
        
        # Calculate final success rate with weighted factors
        success_rate = (base_success * 0.35 +
                       size_factor * 0.25 +
                       market_factor * 0.20 +
                       experience_factor * 0.20)
        
        # Add smaller random variation
        return np.clip(success_rate * random.uniform(0.9, 1.1), 0.3, 0.95)

    def _predict_metrics_with_uncertainty(self,
                                        state: Dict,
                                        scenario_modifiers: Dict,
                                        industry_params: Dict) -> Dict:
        """Predict metrics with more realistic uncertainty and enhanced constraints"""
        predictions = {}
        sentiment_mod = scenario_modifiers['market_sentiment']
        risk_mod = scenario_modifiers['risk_modifier']
        cycle_effect = scenario_modifiers.get('cycle_effect', 0) * 0.85  # Dampened effect
        
        # Calculate profit margin with enhanced constraints
        revenue = state['revenues']
        costs = state['costs']
        profit_margin = ((revenue - costs) / revenue) * 100
        
        # Apply industry constraints with smoother transitions
        profit_margin = np.clip(
            profit_margin,
            industry_params['min_profit_margin'] * 100,
            industry_params['max_profit_margin'] * 100
        )
        
        # Calculate revenue growth with enhanced constraints
        prev_revenue = state.get('prev_revenue', revenue / 1.05)  # More conservative default
        revenue_growth = ((revenue - prev_revenue) / prev_revenue) * 100
        revenue_growth = np.clip(
            revenue_growth,
            industry_params['min_growth_rate'] * 100,
            industry_params['max_growth_rate'] * 100
        )
        
        # Enhanced market value calculation with more realistic factors
        market_value_factors = {
            'profit_margin': 0.25,    # Reduced from 0.3
            'revenue_growth': 0.25,   # Reduced from 0.3
            'market_sentiment': 0.15,  # Reduced from 0.2
            'cycle_effect': 0.15      # Reduced from 0.2
        }
        
        # Calculate market value with dampened volatility
        base_change = (
            (profit_margin / 100) * market_value_factors['profit_margin'] +
            (revenue_growth / 100) * market_value_factors['revenue_growth'] +
            (sentiment_mod - 1) * market_value_factors['market_sentiment'] +
            cycle_effect * market_value_factors['cycle_effect']
        )
        
        # Apply smoother market value changes
        market_value = state['market_value'] * (1 + np.clip(
            base_change * random.uniform(0.97, 1.03),
            -industry_params['volatility'],
            industry_params['volatility']
        ))

        predictions = {
            'market_value': market_value,
            'profit_margin': profit_margin,
            'revenue_growth': revenue_growth,
            'prev_revenue': revenue
        }
        
        return predictions

    def _apply_industry_constraints(self,
                                  predictions: Dict,
                                  industry_params: Dict) -> Dict:
        """Apply enhanced industry-specific constraints to predictions"""
        constrained = predictions.copy()
        
        # Apply smoother profit margin constraints
        constrained['profit_margin'] = np.clip(
            predictions['profit_margin'],
            industry_params['min_profit_margin'] * 100,
            industry_params['max_profit_margin'] * 100
        )
        
        # Apply enhanced revenue growth constraints
        constrained['revenue_growth'] = np.clip(
            predictions['revenue_growth'],
            industry_params['min_growth_rate'] * 100,
            industry_params['max_growth_rate'] * 100
        )
        
        # Apply market value constraints with smoother transitions
        max_mv_change = industry_params['volatility'] * 90  # Reduced from 100
        prev_mv = predictions.get('prev_market_value', predictions['market_value'])
        mv_change_pct = (predictions['market_value'] - prev_mv) / prev_mv * 100
        
        if abs(mv_change_pct) > max_mv_change:
            sign = 1 if mv_change_pct > 0 else -1
            damping_factor = 0.9  # Added damping factor
            constrained['market_value'] = prev_mv * (1 + sign * max_mv_change * damping_factor / 100)
        
        return constrained
    
    

    def _analyze_changes(self,
                        prev_state: Dict,
                        curr_state: Dict,
                        scenario: str,
                        industry_params: Dict) -> str:
        """Generate detailed insights with industry context"""
        changes = []
        industry = curr_state.get('industry', 'Technology')
        
        # Market value analysis
        mv_change = (curr_state['market_value'] - prev_state['market_value']) / prev_state['market_value']
        if abs(mv_change) > 0.05:  # 5% threshold for significance
            market_context = self._get_market_context(mv_change, scenario, industry_params)
            changes.append(
                f"Market value {'increased' if mv_change > 0 else 'decreased'} "
                f"by {abs(mv_change)*100:.1f}% {market_context}"
            )
        
        # Profit margin analysis
        pm_change = curr_state['profit_margin'] - prev_state['profit_margin']
        if abs(pm_change) > 1:  # 1 percentage point threshold
            industry_context = self._get_industry_context(
                curr_state['profit_margin'],
                industry_params['max_profit_margin'] * 100,
                'profit margin'
            )
            changes.append(
                f"Profit margin {'improved' if pm_change > 0 else 'declined'} "
                f"to {curr_state['profit_margin']:.1f}% {industry_context}"
            )
        
        # Revenue growth analysis
        growth_diff = curr_state['revenue_growth'] - prev_state['revenue_growth']
        if abs(growth_diff) > 2:  # 2 percentage point threshold
            industry_context = self._get_industry_context(
                curr_state['revenue_growth'],
                industry_params['max_growth_rate'] * 100,
                'growth rate'
            )
            changes.append(
                f"Revenue growth {'accelerated' if growth_diff > 0 else 'decelerated'} "
                f"to {curr_state['revenue_growth']:.1f}% {industry_context}"
            )
        
        # Employee growth analysis
        emp_change = (curr_state['employees'] - prev_state['employees']) / prev_state['employees']
        if abs(emp_change) > 0.05:  # 5% threshold
            changes.append(
                f"Workforce {'expanded' if emp_change > 0 else 'contracted'} "
                f"by {abs(emp_change)*100:.1f}% to {int(curr_state['employees'])} employees"
            )
        
        return " | ".join(changes) if changes else "Metrics remained relatively stable"

    def _get_market_context(self,
                           change: float,
                           scenario: str,
                           industry_params: Dict) -> str:
        """Generate market context for changes"""
        if scenario == 'optimistic':
            if change > 0:
                return "amid favorable market conditions"
            else:
                return "despite favorable market conditions"
        elif scenario == 'pessimistic':
            if change < 0:
                return "amid challenging market conditions"
            else:
                return "despite market headwinds"
        else:
            if abs(change) > industry_params['volatility']:
                return f"showing unusual volatility for the {industry_params.get('industry', 'industry')}"
            return ""

    def _get_industry_context(self,
                            value: float,
                            max_value: float,
                            metric_name: str) -> str:
        """Generate industry context for metrics"""
        if value > max_value * 0.9:
            return f"(approaching {metric_name} ceiling for the industry)"
        elif value > max_value * 0.7:
            return f"(strong performance for the industry)"
        return ""

