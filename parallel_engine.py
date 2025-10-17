"""
Parallel processing implementations for Monte Carlo simulations
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import List, Callable, Any, Dict
from joblib import Parallel, delayed
import warnings

from monte_carlo_engine import SimulationResult, BaseMonteCarloEngine
from config import CONFIG

def _worker_function(engine_class, engine_params, method_name, num_sims, chunk_id, seed_offset=0):
    """Worker function for parallel processing"""
    try:
        # Create engine instance in worker process
        engine = engine_class(**engine_params)
        
        # Set unique random seed for each worker
        engine.random_state = np.random.RandomState(CONFIG.random_seed + seed_offset + chunk_id)
        
        # Get the method to call
        method = getattr(engine, method_name)
        
        # Run simulations
        result = method(num_sims)
        
        return {
            'paths': result.paths,
            'final_values': result.final_values,
            'execution_time': result.execution_time,
            'chunk_id': chunk_id,
            'method': result.method + '_parallel'
        }
    except Exception as e:
        return {'error': str(e), 'chunk_id': chunk_id}

class ParallelMonteCarloEngine:
    """Parallel processing wrapper for Monte Carlo engines"""
    
    def __init__(self, engine: BaseMonteCarloEngine, config=None):
        self.engine = engine
        self.config = config or CONFIG
        self.engine_class = type(engine)
        
        # Extract engine parameters for serialization
        self.engine_params = self._extract_engine_params()
    
    def _extract_engine_params(self) -> Dict[str, Any]:
        """Extract engine parameters for worker processes"""
        params = {}
        
        # Common parameters
        if hasattr(self.engine, 'S0'):
            params['initial_price'] = self.engine.S0
        if hasattr(self.engine, 'mu'):
            params['drift'] = self.engine.mu
        if hasattr(self.engine, 'sigma'):
            params['volatility'] = self.engine.sigma
        if hasattr(self.engine, 'T'):
            params['time_horizon'] = self.engine.T
        if hasattr(self.engine, 'K'):
            params['strike'] = self.engine.K
        if hasattr(self.engine, 'option_type'):
            params['option_type'] = self.engine.option_type
        if hasattr(self.engine, 'initial_value'):
            params['initial_value'] = self.engine.initial_value
        
        # Add underlying engine for composite engines
        if hasattr(self.engine, 'underlying_engine'):
            params['underlying_engine'] = self.engine.underlying_engine
        if hasattr(self.engine, 'portfolio_engine'):
            params['portfolio_engine'] = self.engine.portfolio_engine
        
        params['config'] = self.config
        return params
    
    def _split_simulations(self, total_simulations: int, num_chunks: int) -> List[int]:
        """Split total simulations into chunks for parallel processing"""
        base_size = total_simulations // num_chunks
        remainder = total_simulations % num_chunks
        
        chunk_sizes = [base_size] * num_chunks
        for i in range(remainder):
            chunk_sizes[i] += 1
        
        return chunk_sizes
    
    def _combine_results(self, worker_results: List[Dict], method: str) -> SimulationResult:
        """Combine results from parallel workers"""
        # Filter out error results
        valid_results = [r for r in worker_results if 'error' not in r]
        error_results = [r for r in worker_results if 'error' in r]
        
        if error_results:
            warnings.warn(f"Some workers failed: {[r['error'] for r in error_results]}")
        
        if not valid_results:
            raise RuntimeError("All parallel workers failed")
        
        # Combine paths and final values
        all_paths = np.vstack([r['paths'] for r in valid_results])
        all_final_values = np.concatenate([r['final_values'] for r in valid_results])
        
        # Calculate total execution time (max of all workers)
        total_time = max([r['execution_time'] for r in valid_results])
        
        return SimulationResult(
            paths=all_paths,
            final_values=all_final_values,
            statistics=None,
            execution_time=total_time,
            method=method
        )
    
    def simulate_multiprocessing(self, num_simulations: int, 
                                method_name: str = "simulate_serial") -> SimulationResult:
        """Run simulations using multiprocessing"""
        start_time = time.time()
        
        num_processes = self.config.num_processes
        chunk_sizes = self._split_simulations(num_simulations, num_processes)
        
        # Create worker arguments
        worker_args = [
            (self.engine_class, self.engine_params, method_name, chunk_size, i)
            for i, chunk_size in enumerate(chunk_sizes)
        ]
        
        # Run parallel processes
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(_worker_function, *args)
                for args in worker_args
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e), 'chunk_id': -1})
        
        total_time = time.time() - start_time
        
        # Combine results
        combined_result = self._combine_results(results, "multiprocessing")
        combined_result.execution_time = total_time
        
        return combined_result
    
    def simulate_concurrent_futures(self, num_simulations: int,
                                   method_name: str = "simulate_serial",
                                   use_threads: bool = False) -> SimulationResult:
        """Run simulations using concurrent.futures"""
        start_time = time.time()
        
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        max_workers = self.config.num_processes
        
        if use_threads:
            # For thread-based execution, use more workers
            max_workers = min(32, (num_simulations // 1000) + 1)
        
        chunk_sizes = self._split_simulations(num_simulations, max_workers)
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {}
            for i, chunk_size in enumerate(chunk_sizes):
                future = executor.submit(
                    _worker_function,
                    self.engine_class,
                    self.engine_params,
                    method_name,
                    chunk_size,
                    i
                )
                future_to_chunk[future] = i
            
            # Collect results
            results = []
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    chunk_id = future_to_chunk[future]
                    results.append({'error': str(e), 'chunk_id': chunk_id})
        
        total_time = time.time() - start_time
        
        # Combine results
        method_name = f"concurrent_{'threads' if use_threads else 'processes'}"
        combined_result = self._combine_results(results, method_name)
        combined_result.execution_time = total_time
        
        return combined_result
    
    def simulate_joblib(self, num_simulations: int,
                       method_name: str = "simulate_serial",
                       backend: str = "multiprocessing") -> SimulationResult:
        """Run simulations using joblib"""
        start_time = time.time()
        
        num_jobs = self.config.num_processes
        chunk_sizes = self._split_simulations(num_simulations, num_jobs)
        
        # Create delayed functions
        delayed_functions = [
            delayed(_worker_function)(
                self.engine_class,
                self.engine_params,
                method_name,
                chunk_size,
                i
            )
            for i, chunk_size in enumerate(chunk_sizes)
        ]
        
        # Execute in parallel
        results = Parallel(n_jobs=num_jobs, backend=backend)(delayed_functions)
        
        total_time = time.time() - start_time
        
        # Combine results
        combined_result = self._combine_results(results, f"joblib_{backend}")
        combined_result.execution_time = total_time
        
        return combined_result

class VectorizedMonteCarloEngine:
    """Highly optimized vectorized Monte Carlo engine"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
    
    def simulate_gbm_vectorized(self, initial_price: float, drift: float,
                               volatility: float, time_horizon: float,
                               num_simulations: int) -> SimulationResult:
        """Highly optimized vectorized GBM simulation"""
        start_time = time.time()
        
        dt = time_horizon / self.config.time_steps
        
        # Generate all random numbers at once
        np.random.seed(self.config.random_seed)
        random_increments = np.random.normal(
            0, np.sqrt(dt),
            (num_simulations, self.config.time_steps)
        )
        
        # Pre-calculate constants
        drift_term = (drift - 0.5 * volatility**2) * dt
        vol_term = volatility
        
        # Initialize paths
        paths = np.zeros((num_simulations, self.config.time_steps + 1))
        paths[:, 0] = initial_price
        
        # Vectorized path generation using cumulative sum for efficiency
        log_returns = drift_term + vol_term * random_increments
        cumulative_returns = np.cumsum(log_returns, axis=1)
        
        # Calculate paths using broadcasting
        paths[:, 1:] = initial_price * np.exp(cumulative_returns)
        
        final_values = paths[:, -1]
        execution_time = time.time() - start_time
        
        return SimulationResult(
            paths=paths,
            final_values=final_values,
            statistics=None,
            execution_time=execution_time,
            method="vectorized_optimized"
        )
    
    def simulate_option_pricing_vectorized(self, initial_price: float, drift: float,
                                         volatility: float, time_horizon: float,
                                         strike: float, option_type: str,
                                         num_simulations: int) -> SimulationResult:
        """Vectorized option pricing simulation"""
        start_time = time.time()
        
        # Use vectorized GBM
        gbm_result = self.simulate_gbm_vectorized(
            initial_price, drift, volatility, time_horizon, num_simulations
        )
        
        final_prices = gbm_result.final_values
        
        # Calculate payoffs vectorized
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - strike, 0)
        elif option_type.lower() == "put":
            payoffs = np.maximum(strike - final_prices, 0)
        else:
            raise ValueError(f"Unsupported option type: {option_type}")
        
        # Discount to present value
        option_values = payoffs * np.exp(-self.config.risk_free_rate * time_horizon)
        
        execution_time = time.time() - start_time
        
        # Calculate statistics
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
            paths=gbm_result.paths,
            final_values=option_values,
            statistics=statistics,
            execution_time=execution_time,
            method="vectorized_option_pricing"
        )

def benchmark_parallel_methods(engine: BaseMonteCarloEngine, 
                              num_simulations: int,
                              methods: List[str] = None) -> Dict[str, SimulationResult]:
    """Benchmark different parallel processing methods"""
    if methods is None:
        methods = ['serial', 'multiprocessing', 'concurrent_futures', 'joblib']
    
    results = {}
    parallel_engine = ParallelMonteCarloEngine(engine)
    
    for method in methods:
        print(f"Running benchmark for {method}...")
        
        try:
            if method == 'serial':
                result = engine.simulate_serial(num_simulations)
            elif method == 'multiprocessing':
                result = parallel_engine.simulate_multiprocessing(num_simulations)
            elif method == 'concurrent_futures':
                result = parallel_engine.simulate_concurrent_futures(num_simulations)
            elif method == 'joblib':
                result = parallel_engine.simulate_joblib(num_simulations)
            elif method == 'vectorized' and hasattr(engine, 'simulate_vectorized'):
                result = engine.simulate_vectorized(num_simulations)
            else:
                print(f"Skipping unsupported method: {method}")
                continue
            
            results[method] = result
            print(f"{method}: {result.execution_time:.3f}s")
            
        except Exception as e:
            print(f"Error in {method}: {e}")
    
    return results