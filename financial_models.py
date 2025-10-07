"""
Advanced financial models for Monte Carlo risk management
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
import time

from monte_carlo_engine import BaseMonteCarloEngine, SimulationResult
from config import CONFIG

@dataclass
class ModelParameters:
    """Container for model parameters"""
    pass

@dataclass
class BlackScholesParameters(ModelParameters):
    """Black-Scholes model parameters"""
    initial_price: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0

@dataclass
class HestonParameters(ModelParameters):
    """Heston stochastic volatility model parameters"""
    initial_price: float
    initial_variance: float
    risk_free_rate: float
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    xi: float     # Volatility of volatility
    rho: float    # Correlation between price and volatility
    dividend_yield: float = 0.0

@dataclass
class JumpDiffusionParameters(ModelParameters):
    """Jump-diffusion (Merton) model parameters"""
    initial_price: float
    risk_free_rate: float
    volatility: float
    jump_intensity: float
    jump_mean: float
    jump_std: float
    dividend_yield: float = 0.0

class FinancialModel(ABC):
    """Abstract base class for financial models"""
    
    def __init__(self, parameters: ModelParameters, config=None):
        self.parameters = parameters
        self.config = config or CONFIG
        self.random_state = np.random.RandomState(self.config.random_seed)
    
    @abstractmethod
    def simulate_path(self, time_horizon: float, time_steps: int = None) -> np.ndarray:
        """Simulate a single price path"""
        pass
    
    @abstractmethod
    def simulate_paths(self, num_paths: int, time_horizon: float, 
                      time_steps: int = None) -> np.ndarray:
        """Simulate multiple price paths"""
        pass
    
    def reset_random_state(self):
        """Reset random state for reproducibility"""
        self.random_state = np.random.RandomState(self.config.random_seed)

class BlackScholesModel(FinancialModel):
    """Black-Scholes geometric Brownian motion model"""
    
    def __init__(self, parameters: BlackScholesParameters, config=None):
        super().__init__(parameters, config)
    
    def simulate_path(self, time_horizon: float, time_steps: int = None) -> np.ndarray:
        """Simulate a single Black-Scholes path"""
        if time_steps is None:
            time_steps = self.config.time_steps
        
        dt = time_horizon / time_steps
        
        # Generate random increments
        dW = self.random_state.normal(0, np.sqrt(dt), time_steps)
        
        # Initialize path
        path = np.zeros(time_steps + 1)
        path[0] = self.parameters.initial_price
        
        # Calculate drift
        drift = (self.parameters.risk_free_rate - 
                self.parameters.dividend_yield - 
                0.5 * self.parameters.volatility**2) * dt
        
        # Generate path
        for i in range(1, time_steps + 1):
            path[i] = path[i-1] * np.exp(
                drift + self.parameters.volatility * dW[i-1]
            )
        
        return path
    
    def simulate_paths(self, num_paths: int, time_horizon: float, 
                      time_steps: int = None) -> np.ndarray:
        """Simulate multiple Black-Scholes paths (vectorized)"""
        if time_steps is None:
            time_steps = self.config.time_steps
        
        dt = time_horizon / time_steps
        
        # Generate all random increments
        dW = self.random_state.normal(0, np.sqrt(dt), (num_paths, time_steps))
        
        # Initialize paths
        paths = np.zeros((num_paths, time_steps + 1))
        paths[:, 0] = self.parameters.initial_price
        
        # Calculate drift
        drift = (self.parameters.risk_free_rate - 
                self.parameters.dividend_yield - 
                0.5 * self.parameters.volatility**2) * dt
        
        # Generate paths vectorized
        for i in range(1, time_steps + 1):
            paths[:, i] = paths[:, i-1] * np.exp(
                drift + self.parameters.volatility * dW[:, i-1]
            )
        
        return paths
    
    def analytical_option_price(self, strike: float, time_to_maturity: float,
                               option_type: str = "call") -> Dict[str, float]:
        """Calculate analytical Black-Scholes option price"""
        S = self.parameters.initial_price
        K = strike
        T = time_to_maturity
        r = self.parameters.risk_free_rate
        q = self.parameters.dividend_yield
        sigma = self.parameters.volatility
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == "call":
            price = (S*np.exp(-q*T)*stats.norm.cdf(d1) - 
                    K*np.exp(-r*T)*stats.norm.cdf(d2))
        else:  # put
            price = (K*np.exp(-r*T)*stats.norm.cdf(-d2) - 
                    S*np.exp(-q*T)*stats.norm.cdf(-d1))
        
        # Calculate Greeks
        delta = np.exp(-q*T) * stats.norm.cdf(d1) if option_type.lower() == "call" else np.exp(-q*T) * (stats.norm.cdf(d1) - 1)
        gamma = np.exp(-q*T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta_factor = (-S*np.exp(-q*T)*stats.norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                       r*K*np.exp(-r*T)*stats.norm.cdf(d2 if option_type.lower() == "call" else -d2))
        theta = theta_factor + q*S*np.exp(-q*T)*stats.norm.cdf(d1 if option_type.lower() == "call" else -d1)
        vega = S * np.exp(-q*T) * stats.norm.pdf(d1) * np.sqrt(T)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

class HestonModel(FinancialModel):
    """Heston stochastic volatility model"""
    
    def __init__(self, parameters: HestonParameters, config=None):
        super().__init__(parameters, config)
        # Ensure Feller condition is satisfied
        if 2 * parameters.kappa * parameters.theta <= parameters.xi**2:
            warnings.warn("Feller condition not satisfied. Variance may become negative.")
    
    def simulate_path(self, time_horizon: float, time_steps: int = None) -> np.ndarray:
        """Simulate a single Heston path"""
        if time_steps is None:
            time_steps = self.config.time_steps
        
        dt = time_horizon / time_steps
        
        # Initialize paths
        price_path = np.zeros(time_steps + 1)
        variance_path = np.zeros(time_steps + 1)
        
        price_path[0] = self.parameters.initial_price
        variance_path[0] = self.parameters.initial_variance
        
        # Generate correlated random numbers
        for i in range(time_steps):
            # Generate independent random numbers
            Z1 = self.random_state.normal()
            Z2 = self.random_state.normal()
            
            # Create correlation
            W1 = Z1
            W2 = self.parameters.rho * Z1 + np.sqrt(1 - self.parameters.rho**2) * Z2
            
            # Update variance (with reflection to ensure non-negativity)
            v_current = max(variance_path[i], 0)
            dv = (self.parameters.kappa * (self.parameters.theta - v_current) * dt + 
                  self.parameters.xi * np.sqrt(v_current * dt) * W2)
            variance_path[i + 1] = max(v_current + dv, 0)
            
            # Update price
            ds = ((self.parameters.risk_free_rate - self.parameters.dividend_yield) * dt + 
                  np.sqrt(v_current * dt) * W1)
            price_path[i + 1] = price_path[i] * np.exp(ds)
        
        return price_path
    
    def simulate_paths(self, num_paths: int, time_horizon: float, 
                      time_steps: int = None) -> np.ndarray:
        """Simulate multiple Heston paths"""
        if time_steps is None:
            time_steps = self.config.time_steps
        
        paths = np.zeros((num_paths, time_steps + 1))
        
        for path_idx in range(num_paths):
            paths[path_idx] = self.simulate_path(time_horizon, time_steps)
        
        return paths

class JumpDiffusionModel(FinancialModel):
    """Merton jump-diffusion model"""
    
    def __init__(self, parameters: JumpDiffusionParameters, config=None):
        super().__init__(parameters, config)
    
    def simulate_path(self, time_horizon: float, time_steps: int = None) -> np.ndarray:
        """Simulate a single jump-diffusion path"""
        if time_steps is None:
            time_steps = self.config.time_steps
        
        dt = time_horizon / time_steps
        
        # Initialize path
        path = np.zeros(time_steps + 1)
        path[0] = self.parameters.initial_price
        
        for i in range(time_steps):
            # Generate Brownian motion increment
            dW = self.random_state.normal(0, np.sqrt(dt))
            
            # Generate jump
            jump_occurred = self.random_state.poisson(self.parameters.jump_intensity * dt)
            jump_size = 0
            if jump_occurred > 0:
                # Multiple jumps can occur in one time step
                for _ in range(jump_occurred):
                    jump_size += self.random_state.normal(
                        self.parameters.jump_mean, 
                        self.parameters.jump_std
                    )
            
            # Calculate drift (adjusted for expected jump size)
            expected_jump = self.parameters.jump_intensity * (
                np.exp(self.parameters.jump_mean + 0.5 * self.parameters.jump_std**2) - 1
            )
            
            drift = (self.parameters.risk_free_rate - 
                    self.parameters.dividend_yield - 
                    0.5 * self.parameters.volatility**2 - 
                    expected_jump) * dt
            
            # Update price
            price_change = (drift + 
                          self.parameters.volatility * dW + 
                          jump_size)
            
            path[i + 1] = path[i] * np.exp(price_change)
        
        return path
    
    def simulate_paths(self, num_paths: int, time_horizon: float, 
                      time_steps: int = None) -> np.ndarray:
        """Simulate multiple jump-diffusion paths"""
        if time_steps is None:
            time_steps = self.config.time_steps
        
        paths = np.zeros((num_paths, time_steps + 1))
        
        for path_idx in range(num_paths):
            paths[path_idx] = self.simulate_path(time_horizon, time_steps)
        
        return paths

class AdvancedVaREngine(BaseMonteCarloEngine):
    """Advanced VaR engine with multiple models and risk measures"""
    
    def __init__(self, model: FinancialModel, portfolio_weights: np.ndarray = None,
                 initial_value: float = 1000000, config=None):
        super().__init__(config)
        self.model = model
        self.portfolio_weights = portfolio_weights or np.array([1.0])
        self.initial_value = initial_value
    
    def generate_path(self, time_horizon: float = 1/252) -> float:
        """Generate P&L for a single path"""
        path = self.model.simulate_path(time_horizon)
        final_value = path[-1]
        return (final_value / self.model.parameters.initial_price - 1) * self.initial_value
    
    def simulate_serial(self, num_simulations: int, time_horizon: float = 1/252) -> SimulationResult:
        """Run VaR simulations"""
        start_time = time.time()
        
        # Generate all paths at once for efficiency
        paths = self.model.simulate_paths(num_simulations, time_horizon)
        
        # Calculate P&L
        returns = paths[:, -1] / paths[:, 0] - 1
        pnl_values = returns * self.initial_value
        
        execution_time = time.time() - start_time
        
        # Calculate comprehensive risk statistics
        statistics = self._calculate_risk_statistics(pnl_values, returns)
        
        return SimulationResult(
            paths=paths,
            final_values=pnl_values,
            statistics=statistics,
            execution_time=execution_time,
            method="advanced_var"
        )
    
    def _calculate_risk_statistics(self, pnl_values: np.ndarray, 
                                 returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive risk statistics"""
        sorted_pnl = np.sort(pnl_values)
        sorted_returns = np.sort(returns)
        
        statistics = {
            # Basic statistics
            'mean_pnl': np.mean(pnl_values),
            'std_pnl': np.std(pnl_values),
            'skewness': stats.skew(pnl_values),
            'kurtosis': stats.kurtosis(pnl_values),
            'min_pnl': np.min(pnl_values),
            'max_pnl': np.max(pnl_values),
            
            # Return statistics
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        }
        
        # VaR calculations
        for confidence_level in self.config.confidence_levels:
            percentile = (1 - confidence_level) * 100
            var_value = -np.percentile(sorted_pnl, percentile)
            statistics[f'VaR_{confidence_level}'] = var_value
            
            # Expected Shortfall (Conditional VaR)
            threshold_idx = int(len(sorted_pnl) * (1 - confidence_level))
            if threshold_idx > 0:
                expected_shortfall = -np.mean(sorted_pnl[:threshold_idx])
                statistics[f'ES_{confidence_level}'] = expected_shortfall
            
            # Return-based VaR
            var_return = -np.percentile(sorted_returns, percentile)
            statistics[f'VaR_return_{confidence_level}'] = var_return
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        statistics['max_drawdown'] = np.min(drawdown)
        
        return statistics

class StressTestEngine:
    """Stress testing engine for scenario analysis"""
    
    def __init__(self, base_model: FinancialModel, config=None):
        self.base_model = base_model
        self.config = config or CONFIG
    
    def create_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Create predefined stress scenarios"""
        scenarios = {
            'market_crash': {
                'volatility_multiplier': 2.0,
                'drift_adjustment': -0.20,  # 20% annual decline
                'description': '2008-style market crash'
            },
            'high_volatility': {
                'volatility_multiplier': 1.5,
                'drift_adjustment': 0.0,
                'description': 'High volatility environment'
            },
            'low_interest_rates': {
                'volatility_multiplier': 1.0,
                'drift_adjustment': -0.02,  # 2% lower drift
                'description': 'Low interest rate environment'
            },
            'black_swan': {
                'volatility_multiplier': 3.0,
                'drift_adjustment': -0.30,  # 30% decline
                'description': 'Black swan event'
            }
        }
        return scenarios
    
    def run_stress_test(self, scenario_params: Dict[str, float], 
                       num_simulations: int = 10000,
                       time_horizon: float = 1/252) -> Dict[str, Any]:
        """Run stress test for given scenario"""
        # Create stressed model parameters
        original_params = self.base_model.parameters
        
        if isinstance(original_params, BlackScholesParameters):
            stressed_params = BlackScholesParameters(
                initial_price=original_params.initial_price,
                risk_free_rate=original_params.risk_free_rate + scenario_params.get('drift_adjustment', 0),
                volatility=original_params.volatility * scenario_params.get('volatility_multiplier', 1),
                dividend_yield=original_params.dividend_yield
            )
            stressed_model = BlackScholesModel(stressed_params, self.config)
        else:
            # For other models, implement similar logic
            stressed_model = self.base_model
        
        # Run stressed simulation
        var_engine = AdvancedVaREngine(stressed_model, config=self.config)
        result = var_engine.simulate_serial(num_simulations, time_horizon)
        
        return {
            'scenario_params': scenario_params,
            'simulation_result': result,
            'stressed_parameters': stressed_params.__dict__ if isinstance(stressed_params, BlackScholesParameters) else None
        }
    
    def run_all_stress_tests(self, num_simulations: int = 10000) -> Dict[str, Dict[str, Any]]:
        """Run all predefined stress tests"""
        scenarios = self.create_stress_scenarios()
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            print(f"Running stress test: {scenario_name}")
            results[scenario_name] = self.run_stress_test(scenario_params, num_simulations)
        
        return results