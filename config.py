"""
Configuration file for Monte Carlo Risk Management System
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimulationConfig:
    """Configuration class for Monte Carlo simulations"""
    
    # Simulation parameters
    num_simulations: int = 100000
    time_steps: int = 252  # Trading days in a year
    dt: float = 1/252  # Daily time step
    
    # Parallel processing
    num_processes: int = None  # None = use all available cores
    chunk_size: int = 10000
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Financial parameters (defaults for testing)
    initial_price: float = 100.0
    risk_free_rate: float = 0.05
    volatility: float = 0.2
    time_to_maturity: float = 1.0
    
    # VaR parameters
    confidence_levels: list = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99, 0.999]
        
        if self.num_processes is None:
            import multiprocessing
            self.num_processes = multiprocessing.cpu_count()

# Global configuration instance
CONFIG = SimulationConfig()

# Market data configuration
MARKET_CONFIG = {
    'trading_days_per_year': 252,
    'hours_per_trading_day': 6.5,
    'business_day_convention': 'following',
    'day_count_convention': 'ACT/365'
}

# Supported models
SUPPORTED_MODELS = {
    'geometric_brownian_motion': 'GBMModel',
    'heston': 'HestonModel',
    'jump_diffusion': 'JumpDiffusionModel',
    'black_scholes': 'BlackScholesModel'
}

# Performance benchmarking settings
BENCHMARK_CONFIG = {
    'warmup_runs': 3,
    'measurement_runs': 10,
    'methods': ['serial', 'multiprocessing', 'concurrent_futures', 'vectorized'],
    'simulation_sizes': [1000, 10000, 100000, 1000000]
}