"""
Performance benchmarking system for Monte Carlo simulations
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import multiprocessing
import psutil
import platform
import warnings
from contextlib import contextmanager
import gc

from config import CONFIG, BENCHMARK_CONFIG
from monte_carlo_engine import GeometricBrownianMotionEngine
from parallel_engine import ParallelMonteCarloEngine, VectorizedMonteCarloEngine, benchmark_parallel_methods
from financial_models import BlackScholesModel, BlackScholesParameters

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    method: str
    num_simulations: int
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_sims_per_second: float
    speedup_ratio: float
    efficiency: float
    accuracy_error: float
    system_info: Dict[str, Any]

@dataclass
class SystemInfo:
    """System information for benchmarking"""
    cpu_count: int
    cpu_freq_mhz: float
    total_memory_gb: float
    available_memory_gb: float
    platform: str
    python_version: str

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.benchmark_config = BENCHMARK_CONFIG
        self.system_info = self._get_system_info()
        self.baseline_results = {}
        
    def _get_system_info(self) -> SystemInfo:
        """Get system information"""
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        
        return SystemInfo(
            cpu_count=multiprocessing.cpu_count(),
            cpu_freq_mhz=cpu_freq.current if cpu_freq else 0,
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            platform=platform.platform(),
            python_version=platform.python_version()
        )
    
    @contextmanager
    def _monitor_resources(self):
        """Context manager to monitor CPU and memory usage"""
        process = psutil.Process()
        
        # Initial measurements
        cpu_percent_start = process.cpu_percent()
        memory_start = process.memory_info().rss / (1024**2)  # MB
        
        start_time = time.time()
        yield
        end_time = time.time()
        
        # Final measurements
        cpu_percent_end = process.cpu_percent()
        memory_end = process.memory_info().rss / (1024**2)  # MB
        
        # Store measurements
        self._last_cpu_usage = (cpu_percent_start + cpu_percent_end) / 2
        self._last_memory_usage = max(memory_start, memory_end)
        self._last_execution_time = end_time - start_time
    
    def _calculate_accuracy_error(self, result_values: np.ndarray, 
                                 analytical_value: float = None) -> float:
        """Calculate accuracy error compared to analytical solution"""
        if analytical_value is None:
            # Use the mean as reference (for relative comparison)
            return 0.0
        
        estimated_value = np.mean(result_values)
        relative_error = abs(estimated_value - analytical_value) / abs(analytical_value)
        return relative_error * 100  # Return as percentage
    
    def _warm_up_system(self, engine, num_warmup_sims: int = 1000):
        """Warm up the system with small simulations"""
        for _ in range(self.benchmark_config['warmup_runs']):
            try:
                _ = engine.simulate_serial(num_warmup_sims)
                gc.collect()  # Force garbage collection
            except Exception as e:
                warnings.warn(f"Warmup failed: {e}")
    
    def benchmark_single_method(self, engine, method_name: str, 
                               num_simulations: int,
                               analytical_reference: float = None) -> BenchmarkResult:
        """Benchmark a single method"""
        
        # Warm up
        self._warm_up_system(engine, min(1000, num_simulations // 10))
        
        execution_times = []
        results_values = []
        
        # Run multiple measurements
        for _ in range(self.benchmark_config['measurement_runs']):
            with self._monitor_resources():
                try:
                    if method_name == 'serial':
                        result = engine.simulate_serial(num_simulations)
                    elif method_name == 'vectorized' and hasattr(engine, 'simulate_vectorized'):
                        result = engine.simulate_vectorized(num_simulations)
                    elif method_name in ['multiprocessing', 'concurrent_futures', 'joblib']:
                        parallel_engine = ParallelMonteCarloEngine(engine, self.config)
                        if method_name == 'multiprocessing':
                            result = parallel_engine.simulate_multiprocessing(num_simulations)
                        elif method_name == 'concurrent_futures':
                            result = parallel_engine.simulate_concurrent_futures(num_simulations)
                        elif method_name == 'joblib':
                            result = parallel_engine.simulate_joblib(num_simulations)
                    else:
                        raise ValueError(f"Unknown method: {method_name}")
                    
                    execution_times.append(self._last_execution_time)
                    results_values.append(result.final_values)
                    
                except Exception as e:
                    warnings.warn(f"Benchmark failed for {method_name}: {e}")
                    return None
                
                # Clean up
                gc.collect()
                time.sleep(0.1)  # Brief pause between runs
        
        # Calculate statistics
        avg_execution_time = np.mean(execution_times)
        throughput = num_simulations / avg_execution_time
        
        # Calculate speedup (compared to serial if available)
        speedup_ratio = 1.0
        efficiency = 1.0
        if 'serial' in self.baseline_results:
            speedup_ratio = self.baseline_results['serial'].execution_time / avg_execution_time
            efficiency = speedup_ratio / self.system_info.cpu_count
        
        # Calculate accuracy
        all_values = np.concatenate(results_values)
        accuracy_error = self._calculate_accuracy_error(all_values, analytical_reference)
        
        return BenchmarkResult(
            method=method_name,
            num_simulations=num_simulations,
            execution_time=avg_execution_time,
            memory_usage_mb=self._last_memory_usage,
            cpu_usage_percent=self._last_cpu_usage,
            throughput_sims_per_second=throughput,
            speedup_ratio=speedup_ratio,
            efficiency=efficiency,
            accuracy_error=accuracy_error,
            system_info=asdict(self.system_info)
        )
    
    def run_comprehensive_benchmark(self, 
                                  simulation_sizes: List[int] = None,
                                  methods: List[str] = None,
                                  model_params: Dict[str, Any] = None) -> pd.DataFrame:
        """Run comprehensive benchmark across multiple methods and simulation sizes"""
        
        if simulation_sizes is None:
            simulation_sizes = self.benchmark_config['simulation_sizes']
        
        if methods is None:
            methods = self.benchmark_config['methods']
        
        # Default model parameters
        if model_params is None:
            model_params = {
                'initial_price': 100.0,
                'drift': 0.05,
                'volatility': 0.2,
                'time_horizon': 1.0
            }
        
        results = []
        
        for size in simulation_sizes:
            print(f"\nBenchmarking with {size:,} simulations...")
            
            # Create fresh engine for each size
            engine = GeometricBrownianMotionEngine(
                initial_price=model_params['initial_price'],
                drift=model_params['drift'],
                volatility=model_params['volatility'],
                time_horizon=model_params['time_horizon'],
                config=self.config
            )
            
            # Calculate analytical reference for accuracy comparison
            bs_params = BlackScholesParameters(
                initial_price=model_params['initial_price'],
                risk_free_rate=model_params['drift'],
                volatility=model_params['volatility']
            )
            bs_model = BlackScholesModel(bs_params)
            
            # Reset baseline for each size
            self.baseline_results = {}
            
            for method in methods:
                print(f"  Testing {method}...")
                
                try:
                    # Skip vectorized if not available
                    if method == 'vectorized' and not hasattr(engine, 'simulate_vectorized'):
                        print(f"    Skipping {method} (not available)")
                        continue
                    
                    result = self.benchmark_single_method(
                        engine=engine,
                        method_name=method,
                        num_simulations=size,
                        analytical_reference=model_params['initial_price']  # Simple reference
                    )
                    
                    if result is not None:
                        # Store baseline (first successful result, usually serial)
                        if method == 'serial':
                            self.baseline_results['serial'] = result
                        
                        results.append(asdict(result))
                        print(f"    {method}: {result.execution_time:.3f}s "
                              f"({result.throughput_sims_per_second:,.0f} sims/s, "
                              f"{result.speedup_ratio:.2f}x speedup)")
                    else:
                        print(f"    {method}: FAILED")
                        
                except Exception as e:
                    print(f"    {method}: ERROR - {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        return df
    
    def benchmark_scalability(self, base_simulations: int = 10000,
                            scale_factors: List[float] = None) -> pd.DataFrame:
        """Benchmark scalability across different simulation sizes"""
        
        if scale_factors is None:
            scale_factors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        simulation_sizes = [int(base_simulations * factor) for factor in scale_factors]
        
        return self.run_comprehensive_benchmark(
            simulation_sizes=simulation_sizes,
            methods=['serial', 'multiprocessing', 'vectorized']
        )
    
    def benchmark_parallel_efficiency(self, num_simulations: int = 100000,
                                    process_counts: List[int] = None) -> pd.DataFrame:
        """Benchmark parallel efficiency with different numbers of processes"""
        
        if process_counts is None:
            max_processes = min(multiprocessing.cpu_count(), 8)
            process_counts = list(range(1, max_processes + 1))
        
        results = []
        original_processes = self.config.num_processes
        
        try:
            for num_processes in process_counts:
                print(f"\nTesting with {num_processes} processes...")
                
                # Update config
                self.config.num_processes = num_processes
                
                # Create engine
                engine = GeometricBrownianMotionEngine(
                    initial_price=100.0,
                    drift=0.05,
                    volatility=0.2,
                    time_horizon=1.0,
                    config=self.config
                )
                
                # Benchmark multiprocessing method
                result = self.benchmark_single_method(
                    engine=engine,
                    method_name='multiprocessing',
                    num_simulations=num_simulations
                )
                
                if result is not None:
                    result_dict = asdict(result)
                    result_dict['num_processes'] = num_processes
                    results.append(result_dict)
                    
                    print(f"  {num_processes} processes: {result.execution_time:.3f}s "
                          f"({result.speedup_ratio:.2f}x speedup, "
                          f"{result.efficiency:.2f} efficiency)")
        
        finally:
            # Restore original configuration
            self.config.num_processes = original_processes
        
        return pd.DataFrame(results)
    
    def generate_benchmark_report(self, results_df: pd.DataFrame) -> str:
        """Generate a comprehensive benchmark report"""
        
        report_lines = []
        report_lines.append("=== Monte Carlo Simulation Performance Benchmark Report ===\n")
        
        # System information
        report_lines.append("SYSTEM INFORMATION:")
        report_lines.append(f"  CPU Cores: {self.system_info.cpu_count}")
        report_lines.append(f"  CPU Frequency: {self.system_info.cpu_freq_mhz:.0f} MHz")
        report_lines.append(f"  Total Memory: {self.system_info.total_memory_gb:.1f} GB")
        report_lines.append(f"  Platform: {self.system_info.platform}")
        report_lines.append("")
        
        if len(results_df) == 0:
            report_lines.append("No benchmark results available.")
            return "\n".join(report_lines)
        
        # Overall performance summary
        report_lines.append("PERFORMANCE SUMMARY:")
        
        # Best performing method for each simulation size
        for size in results_df['num_simulations'].unique():
            size_results = results_df[results_df['num_simulations'] == size]
            best_method = size_results.loc[size_results['throughput_sims_per_second'].idxmax()]
            
            report_lines.append(f"  {size:,} simulations - Best: {best_method['method']} "
                              f"({best_method['throughput_sims_per_second']:,.0f} sims/s, "
                              f"{best_method['speedup_ratio']:.2f}x speedup)")
        
        report_lines.append("")
        
        # Method comparison
        report_lines.append("METHOD COMPARISON (Average across all sizes):")
        method_summary = results_df.groupby('method').agg({
            'execution_time': 'mean',
            'throughput_sims_per_second': 'mean',
            'speedup_ratio': 'mean',
            'efficiency': 'mean',
            'memory_usage_mb': 'mean'
        }).round(3)
        
        for method, stats in method_summary.iterrows():
            report_lines.append(f"  {method:15s}: "
                              f"{stats['throughput_sims_per_second']:8.0f} sims/s | "
                              f"{stats['speedup_ratio']:5.2f}x speedup | "
                              f"{stats['efficiency']:5.2f} efficiency | "
                              f"{stats['memory_usage_mb']:6.1f} MB")
        
        report_lines.append("")
        
        # Scalability analysis
        if len(results_df['num_simulations'].unique()) > 1:
            report_lines.append("SCALABILITY ANALYSIS:")
            
            for method in results_df['method'].unique():
                method_data = results_df[results_df['method'] == method].sort_values('num_simulations')
                if len(method_data) > 1:
                    # Calculate scalability coefficient (how throughput changes with size)
                    sizes = method_data['num_simulations'].values
                    throughputs = method_data['throughput_sims_per_second'].values
                    
                    # Simple linear regression to measure scalability
                    slope = np.corrcoef(np.log(sizes), np.log(throughputs))[0, 1]
                    
                    scalability = "Good" if slope > 0.5 else "Moderate" if slope > 0 else "Poor"
                    report_lines.append(f"  {method:15s}: {scalability:8s} (correlation: {slope:.3f})")
        
        return "\n".join(report_lines)
    
    def save_results(self, results_df: pd.DataFrame, filename_prefix: str = "benchmark"):
        """Save benchmark results to CSV and generate report"""
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"Benchmark results saved to: {csv_filename}")
        
        # Save report
        report = self.generate_benchmark_report(results_df)
        report_filename = f"{filename_prefix}_report_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"Benchmark report saved to: {report_filename}")
        
        return csv_filename, report_filename

def run_quick_benchmark() -> pd.DataFrame:
    """Run a quick benchmark with default settings"""
    benchmark = PerformanceBenchmark()
    
    # Quick test with smaller simulation sizes
    quick_sizes = [1000, 10000, 50000]
    quick_methods = ['serial', 'multiprocessing', 'vectorized']
    
    results = benchmark.run_comprehensive_benchmark(
        simulation_sizes=quick_sizes,
        methods=quick_methods
    )
    
    # Print quick summary
    print("\n" + benchmark.generate_benchmark_report(results))
    
    return results

def run_full_benchmark() -> pd.DataFrame:
    """Run a comprehensive benchmark with all methods and sizes"""
    benchmark = PerformanceBenchmark()
    
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    benchmark.save_results(results, "monte_carlo_full_benchmark")
    
    return results

if __name__ == "__main__":
    print("Running Monte Carlo Simulation Performance Benchmark...")
    
    # Run quick benchmark by default
    results = run_quick_benchmark()
    
    # Ask if user wants full benchmark
    print(f"\nBenchmark completed with {len(results)} test cases.")
    print("Results summary:")
    print(results.groupby('method')['throughput_sims_per_second'].mean().round(0).to_string())