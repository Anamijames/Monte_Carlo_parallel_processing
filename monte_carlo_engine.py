"""
Core Monte Carlo simulation engine for financial risk management
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from config import CONFIG

@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results"""
    paths: np.ndarray
    final_values: np.ndarray
    statistics: Dict
    execution_time: float
    method: str
    
    def __post_init__(self):
        """Calculate basic statistics after initialization"""
        if self.statistics is None:
            self.statistics = {}
        
        self.statistics.update({
            'mean': np.mean(self.final_values),
            'std': np.std(self.final_values),
            'min': np.min(self.final_values),
            'max': np.max(self.final_values),
            'median': np.median(self.final_values),
            'num_simulations': len(self.final_values)
        })

class BaseMonteCarloEngine(ABC):
    """Abstract base class for Monte Carlo simulation engines"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.random_state = np.random.RandomState(self.config.random_seed)
    
    @abstractmethod
    def generate_path(self, *args, **kwargs) -> np.ndarray:
        """Generate a single simulation path"""
        pass
    
    @abstractmethod
    def simulate_serial(self, num_simulations: int, *args, **kwargs) -> SimulationResult:
        """Run simulations serially"""
        pass
    
    def reset_random_state(self):
        """Reset random state for reproducibility"""
        self.random_state = np.random.RandomState(self.config.random_seed)

class GeometricBrownianMotionEngine(BaseMonteCarloEngine):
    """Monte Carlo engine for Geometric Brownian Motion"""
    
    def __init__(self, initial_price: float, drift: float, volatility: float, 
                 time_horizon: float, config=None):
        super().__init__(config)
        self.S0 = initial_price
        self.mu = drift
        self.sigma = volatility
        self.T = time_horizon
        self.dt = self.T / self.config.time_steps
    
    def generate_path(self, random_state=None) -> np.ndarray:
        """Generate a single GBM path"""
        if random_state is None:
            random_state = self.random_state
        
        # Generate random increments
        dW = random_state.normal(0, np.sqrt(self.dt), self.config.time_steps)
        
        # Initialize path
        path = np.zeros(self.config.time_steps + 1)
        path[0] = self.S0
        
        # Generate path using GBM formula
        for i in range(1, self.config.time_steps + 1):
            path[i] = path[i-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW[i-1]
            )
        
        return path
    
    def generate_paths_vectorized(self, num_simulations: int) -> np.ndarray:
        """Generate multiple paths using vectorized operations"""
        # Generate all random increments at once
        dW = self.random_state.normal(
            0, np.sqrt(self.dt), 
            (num_simulations, self.config.time_steps)
        )
        
        # Initialize paths array
        paths = np.zeros((num_simulations, self.config.time_steps + 1))
        paths[:, 0] = self.S0
        
        # Vectorized path generation
        for i in range(1, self.config.time_steps + 1):
            paths[:, i] = paths[:, i-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW[:, i-1]
            )
        
        return paths
    
    def simulate_serial(self, num_simulations: int) -> SimulationResult:
        """Run GBM simulations serially"""
        start_time = time.time()
        
        paths = []
        for _ in range(num_simulations):
            path = self.generate_path()
            paths.append(path)
        
        paths = np.array(paths)
        final_values = paths[:, -1]
        
        execution_time = time.time() - start_time
        
        return SimulationResult(
            paths=paths,
            final_values=final_values,
            statistics=None,
            execution_time=execution_time,
            method="serial"
        )
    
    def simulate_vectorized(self, num_simulations: int) -> SimulationResult:
        """Run GBM simulations using vectorized operations"""
        start_time = time.time()
        
        paths = self.generate_paths_vectorized(num_simulations)
        final_values = paths[:, -1]
        
        execution_time = time.time() - start_time
        
        return SimulationResult(
            paths=paths,
            final_values=final_values,
            statistics=None,
            execution_time=execution_time,
            method="vectorized"
        )

class OptionPricingEngine(BaseMonteCarloEngine):
    """Monte Carlo engine for option pricing"""
    
    def __init__(self, underlying_engine: BaseMonteCarloEngine, 
                 strike: float, option_type: str = "call", config=None):
        super().__init__(config)
        self.underlying_engine = underlying_engine
        self.K = strike
        self.option_type = option_type.lower()
        self.r = self.config.risk_free_rate
        self.T = underlying_engine.T
    
    def payoff(self, final_prices: np.ndarray) -> np.ndarray:
        """Calculate option payoff"""
        if self.option_type == "call":
            return np.maximum(final_prices - self.K, 0)
        elif self.option_type == "put":
            return np.maximum(self.K - final_prices, 0)
        else:
            raise ValueError(f"Unsupported option type: {self.option_type}")
    
    def generate_path(self, random_state=None) -> float:
        """Generate option value for a single path"""
        underlying_path = self.underlying_engine.generate_path(random_state)
        final_price = underlying_path[-1]
        option_payoff = self.payoff(np.array([final_price]))[0]
        # Discount to present value
        return option_payoff * np.exp(-self.r * self.T)
    
    def simulate_serial(self, num_simulations: int) -> SimulationResult:
        """Run option pricing simulations serially"""
        start_time = time.time()
        
        option_values = []
        paths = []
        
        for _ in range(num_simulations):
            underlying_path = self.underlying_engine.generate_path()
            final_price = underlying_path[-1]
            option_payoff = self.payoff(np.array([final_price]))[0]
            option_value = option_payoff * np.exp(-self.r * self.T)
            
            option_values.append(option_value)
            paths.append(underlying_path)
        
        paths = np.array(paths)
        option_values = np.array(option_values)
        
        execution_time = time.time() - start_time
        
        # Calculate option price statistics
        option_price = np.mean(option_values)
        option_std = np.std(option_values)
        
        statistics = {
            'option_price': option_price,
            'option_std_error': option_std / np.sqrt(num_simulations),
            'confidence_interval_95': (
                option_price - 1.96 * option_std / np.sqrt(num_simulations),
                option_price + 1.96 * option_std / np.sqrt(num_simulations)
            )
        }
        
        return SimulationResult(
            paths=paths,
            final_values=option_values,
            statistics=statistics,
            execution_time=execution_time,
            method="serial"
        )

class VaREngine(BaseMonteCarloEngine):
    """Value at Risk calculation engine"""
    
    def __init__(self, portfolio_engine: BaseMonteCarloEngine, 
                 initial_value: float, config=None):
        super().__init__(config)
        self.portfolio_engine = portfolio_engine
        self.initial_value = initial_value
    
    def generate_path(self, random_state=None) -> float:
        """Generate portfolio P&L for a single path"""
        portfolio_path = self.portfolio_engine.generate_path(random_state)
        final_value = portfolio_path[-1]
        return final_value - self.initial_value  # P&L
    
    def calculate_var(self, pnl_values: np.ndarray, 
                     confidence_levels: List[float] = None) -> Dict[str, float]:
        """Calculate VaR at different confidence levels"""
        if confidence_levels is None:
            confidence_levels = self.config.confidence_levels
        
        var_results = {}
        sorted_pnl = np.sort(pnl_values)
        
        for confidence_level in confidence_levels:
            # VaR is the negative of the percentile (loss is negative)
            percentile = (1 - confidence_level) * 100
            var_value = -np.percentile(sorted_pnl, percentile)
            var_results[f'VaR_{confidence_level}'] = var_value
            
            # Expected Shortfall (Conditional VaR)
            threshold_idx = int(len(sorted_pnl) * (1 - confidence_level))
            if threshold_idx > 0:
                expected_shortfall = -np.mean(sorted_pnl[:threshold_idx])
                var_results[f'ES_{confidence_level}'] = expected_shortfall
        
        return var_results
    
    def simulate_serial(self, num_simulations: int) -> SimulationResult:
        """Run VaR simulations serially"""
        start_time = time.time()
        
        pnl_values = []
        paths = []
        
        for _ in range(num_simulations):
            portfolio_path = self.portfolio_engine.generate_path()
            pnl = portfolio_path[-1] - self.initial_value
            
            pnl_values.append(pnl)
            paths.append(portfolio_path)
        
        paths = np.array(paths)
        pnl_values = np.array(pnl_values)
        
        execution_time = time.time() - start_time
        
        # Calculate VaR statistics
        var_statistics = self.calculate_var(pnl_values)
        var_statistics.update({
            'mean_pnl': np.mean(pnl_values),
            'std_pnl': np.std(pnl_values),
            'min_pnl': np.min(pnl_values),
            'max_pnl': np.max(pnl_values)
        })
        
        return SimulationResult(
            paths=paths,
            final_values=pnl_values,
            statistics=var_statistics,
            execution_time=execution_time,
            method="serial"
        )