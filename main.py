"""
Main demonstration script for Monte Carlo Risk Management System

This script demonstrates the key capabilities of the parallel Monte Carlo simulation system:
1. Asset price modeling with geometric Brownian motion
2. Option pricing with Monte Carlo methods
3. Value at Risk (VaR) calculations
4. Parallel processing performance comparison
5. Stress testing and scenario analysis
6. Visualization and reporting

Usage:
    python main.py [--quick] [--full-benchmark] [--stress-test] [--no-plots]
"""

import sys
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our modules
from config import CONFIG, SimulationConfig
from monte_carlo_engine import GeometricBrownianMotionEngine, OptionPricingEngine
from parallel_engine import ParallelMonteCarloEngine, VectorizedMonteCarloEngine, benchmark_parallel_methods
from financial_models import (BlackScholesModel, BlackScholesParameters, 
                             HestonModel, HestonParameters,
                             JumpDiffusionModel, JumpDiffusionParameters,
                             AdvancedVaREngine, StressTestEngine)
from benchmark import PerformanceBenchmark, run_quick_benchmark
from visualization import MonteCarloVisualizer, plot_stress_test_results

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def demonstrate_basic_monte_carlo():
    """Demonstrate basic Monte Carlo simulation"""
    print("=" * 60)
    print("BASIC MONTE CARLO SIMULATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple GBM engine
    engine = GeometricBrownianMotionEngine(
        initial_price=100.0,
        drift=0.05,  # 5% annual drift
        volatility=0.2,  # 20% annual volatility
        time_horizon=1.0,  # 1 year
        config=SimulationConfig(num_simulations=10000, random_seed=42)
    )
    
    print(f"Simulating stock price evolution...")
    print(f"  Initial price: ${engine.S0:.2f}")
    print(f"  Drift: {engine.mu:.1%}")
    print(f"  Volatility: {engine.sigma:.1%}")
    print(f"  Time horizon: {engine.T:.1f} year(s)")
    
    # Run serial simulation
    print(f"\nRunning {CONFIG.num_simulations:,} simulations...")
    result = engine.simulate_serial(CONFIG.num_simulations)
    
    print(f"\nResults:")
    print(f"  Execution time: {result.execution_time:.3f} seconds")
    print(f"  Final price statistics:")
    print(f"    Mean: ${result.statistics['mean']:.2f}")
    print(f"    Std Dev: ${result.statistics['std']:.2f}")
    print(f"    Min: ${result.statistics['min']:.2f}")
    print(f"    Max: ${result.statistics['max']:.2f}")
    print(f"    Median: ${result.statistics['median']:.2f}")
    
    return result

def demonstrate_option_pricing():
    """Demonstrate option pricing with Monte Carlo"""
    print("\n" + "=" * 60)
    print("OPTION PRICING DEMONSTRATION")
    print("=" * 60)
    
    # Create underlying asset engine
    underlying_engine = GeometricBrownianMotionEngine(
        initial_price=100.0,
        drift=0.05,
        volatility=0.2,
        time_horizon=0.25,  # 3 months to expiration
        config=SimulationConfig(num_simulations=50000, random_seed=42)
    )
    
    # Create option pricing engine
    strike_price = 105.0
    option_engine = OptionPricingEngine(
        underlying_engine=underlying_engine,
        strike=strike_price,
        option_type="call"
    )
    
    print(f"Pricing European Call Option:")
    print(f"  Underlying price: ${underlying_engine.S0:.2f}")
    print(f"  Strike price: ${strike_price:.2f}")
    print(f"  Time to expiration: {underlying_engine.T:.2f} years")
    print(f"  Risk-free rate: {CONFIG.risk_free_rate:.1%}")
    print(f"  Volatility: {underlying_engine.sigma:.1%}")
    
    # Price the option
    print(f"\nRunning Monte Carlo option pricing...")
    option_result = option_engine.simulate_serial(50000)
    
    print(f"\nMonte Carlo Results:")
    print(f"  Option price: ${option_result.statistics['option_price']:.4f}")
    print(f"  Standard error: ${option_result.statistics['option_std_error']:.4f}")
    print(f"  95% CI: [${option_result.statistics['confidence_interval_95'][0]:.4f}, "
          f"${option_result.statistics['confidence_interval_95'][1]:.4f}]")
    print(f"  Execution time: {option_result.execution_time:.3f} seconds")
    
    # Compare with analytical Black-Scholes price
    bs_params = BlackScholesParameters(
        initial_price=underlying_engine.S0,
        risk_free_rate=CONFIG.risk_free_rate,
        volatility=underlying_engine.sigma
    )
    bs_model = BlackScholesModel(bs_params)
    analytical_result = bs_model.analytical_option_price(
        strike_price, underlying_engine.T, "call"
    )
    
    print(f"\nAnalytical Black-Scholes Results:")
    print(f"  Option price: ${analytical_result['price']:.4f}")
    print(f"  Delta: {analytical_result['delta']:.4f}")
    print(f"  Gamma: {analytical_result['gamma']:.4f}")
    print(f"  Vega: {analytical_result['vega']:.4f}")
    
    error = abs(option_result.statistics['option_price'] - analytical_result['price'])
    error_pct = error / analytical_result['price'] * 100
    print(f"\nAccuracy:")
    print(f"  Absolute error: ${error:.4f}")
    print(f"  Relative error: {error_pct:.2f}%")
    
    return option_result, analytical_result

def demonstrate_var_calculation():
    """Demonstrate Value at Risk calculation"""
    print("\n" + "=" * 60)
    print("VALUE AT RISK (VAR) DEMONSTRATION")
    print("=" * 60)
    
    # Create a portfolio model
    portfolio_params = BlackScholesParameters(
        initial_price=1000000,  # $1M portfolio
        risk_free_rate=0.02,
        volatility=0.15  # 15% portfolio volatility
    )
    portfolio_model = BlackScholesModel(portfolio_params)
    
    # Create VaR engine
    var_engine = AdvancedVaREngine(
        model=portfolio_model,
        initial_value=1000000
    )
    
    print(f"Calculating VaR for $1M portfolio:")
    print(f"  Portfolio volatility: {portfolio_params.volatility:.1%}")
    print(f"  Time horizon: 1 day (1/252 years)")
    print(f"  Confidence levels: {[f'{cl:.1%}' for cl in CONFIG.confidence_levels]}")
    
    # Calculate VaR
    print(f"\nRunning VaR simulation...")
    var_result = var_engine.simulate_serial(100000, time_horizon=1/252)
    
    print(f"\nVaR Results:")
    for cl in CONFIG.confidence_levels:
        var_value = var_result.statistics[f'VaR_{cl}']
        es_value = var_result.statistics[f'ES_{cl}']
        print(f"  VaR {cl:.1%}: ${var_value:,.0f}")
        print(f"  Expected Shortfall {cl:.1%}: ${es_value:,.0f}")
    
    print(f"\nPortfolio Statistics:")
    print(f"  Mean P&L: ${var_result.statistics['mean_pnl']:,.0f}")
    print(f"  P&L Volatility: ${var_result.statistics['std_pnl']:,.0f}")
    print(f"  Skewness: {var_result.statistics['skewness']:.3f}")
    print(f"  Kurtosis: {var_result.statistics['kurtosis']:.3f}")
    print(f"  Maximum Drawdown: {var_result.statistics['max_drawdown']:.2%}")
    
    return var_result

def demonstrate_parallel_processing():
    """Demonstrate parallel processing capabilities"""
    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Create engine for performance testing
    engine = GeometricBrownianMotionEngine(
        initial_price=100.0,
        drift=0.05,
        volatility=0.2,
        time_horizon=1.0,
        config=SimulationConfig(num_simulations=50000, random_seed=42)
    )
    
    print(f"Comparing serial vs parallel performance:")
    print(f"  Number of simulations: {50000:,}")
    print(f"  Available CPU cores: {CONFIG.num_processes}")
    
    # Test different methods
    methods_to_test = ['serial', 'vectorized', 'multiprocessing', 'concurrent_futures']
    results = {}
    
    for method in methods_to_test:
        print(f"\n  Testing {method}...")
        start_time = time.time()
        
        try:
            if method == 'serial':
                result = engine.simulate_serial(50000)
            elif method == 'vectorized':
                result = engine.simulate_vectorized(50000)
            elif method == 'multiprocessing':
                parallel_engine = ParallelMonteCarloEngine(engine)
                result = parallel_engine.simulate_multiprocessing(50000)
            elif method == 'concurrent_futures':
                parallel_engine = ParallelMonteCarloEngine(engine)
                result = parallel_engine.simulate_concurrent_futures(50000)
            
            results[method] = result
            throughput = 50000 / result.execution_time
            print(f"    Execution time: {result.execution_time:.3f}s")
            print(f"    Throughput: {throughput:,.0f} simulations/second")
            
        except Exception as e:
            print(f"    Failed: {e}")
    
    # Calculate speedups
    if 'serial' in results:
        baseline_time = results['serial'].execution_time
        print(f"\nSpeedup Analysis (vs serial):")
        
        for method, result in results.items():
            if method != 'serial':
                speedup = baseline_time / result.execution_time
                efficiency = speedup / CONFIG.num_processes
                print(f"  {method:18s}: {speedup:.2f}x speedup, {efficiency:.2f} efficiency")
    
    return results

def demonstrate_stress_testing():
    """Demonstrate stress testing capabilities"""
    print("\n" + "=" * 60)
    print("STRESS TESTING DEMONSTRATION")
    print("=" * 60)
    
    # Create base model
    base_params = BlackScholesParameters(
        initial_price=1000000,
        risk_free_rate=0.03,
        volatility=0.20
    )
    base_model = BlackScholesModel(base_params)
    
    # Create stress test engine
    stress_engine = StressTestEngine(base_model)
    
    print(f"Running stress tests on $1M portfolio...")
    print(f"Base parameters:")
    print(f"  Risk-free rate: {base_params.risk_free_rate:.1%}")
    print(f"  Volatility: {base_params.volatility:.1%}")
    
    # Run stress tests
    print(f"\nExecuting predefined stress scenarios...")
    stress_results = stress_engine.run_all_stress_tests(num_simulations=25000)
    
    print(f"\nStress Test Results:")
    for scenario_name, scenario_result in stress_results.items():
        result = scenario_result['simulation_result']
        params = scenario_result['scenario_params']
        
        print(f"\n  {scenario_name.upper()}:")
        print(f"    Description: {params.get('description', 'N/A')}")
        print(f"    VaR 95%: ${result.statistics.get('VaR_0.95', 0):,.0f}")
        print(f"    VaR 99%: ${result.statistics.get('VaR_0.99', 0):,.0f}")
        print(f"    Expected Shortfall 95%: ${result.statistics.get('ES_0.95', 0):,.0f}")
        print(f"    Worst case loss: ${result.statistics.get('min_pnl', 0):,.0f}")
    
    return stress_results

def create_visualizations(results_dict, benchmark_data=None, stress_results=None, save_plots=True):
    """Create and display visualizations"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    visualizer = MonteCarloVisualizer()
    
    # Plot simulation paths for the first result
    if results_dict:
        first_result = list(results_dict.values())[0]
        print("Creating simulation paths plot...")
        
        fig1 = visualizer.plot_simulation_paths(
            first_result, 
            num_paths_to_plot=50,
            title="Sample Monte Carlo Paths"
        )
        
        if save_plots:
            fig1.savefig("monte_carlo_paths.png", dpi=300, bbox_inches='tight')
            print("  Saved: monte_carlo_paths.png")
        
        # Plot final value distribution
        print("Creating final value distribution plot...")
        fig2 = visualizer.plot_final_value_distribution(
            first_result,
            title="Final Value Distribution"
        )
        
        if save_plots:
            fig2.savefig("final_value_distribution.png", dpi=300, bbox_inches='tight')
            print("  Saved: final_value_distribution.png")
    
    # Plot performance comparison if benchmark data available
    if benchmark_data is not None and len(benchmark_data) > 0:
        print("Creating performance comparison plot...")
        fig3 = visualizer.plot_performance_comparison(
            benchmark_data,
            title="Monte Carlo Performance Comparison"
        )
        
        if save_plots:
            fig3.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')
            print("  Saved: performance_comparison.png")
    
    # Plot stress test results if available
    if stress_results is not None:
        print("Creating stress test results plot...")
        fig4 = plot_stress_test_results(
            stress_results,
            title="Portfolio Stress Test Analysis"
        )
        
        if save_plots:
            fig4.savefig("stress_test_results.png", dpi=300, bbox_inches='tight')
            print("  Saved: stress_test_results.png")
    
    # Generate comprehensive HTML report
    print("Generating comprehensive report...")
    report_html = visualizer.generate_comprehensive_report(
        results_dict, 
        benchmark_data, 
        save_path="monte_carlo_report.html"
    )
    
    print("  Saved: monte_carlo_report.html")
    
    return True

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="Monte Carlo Risk Management Demonstration")
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick demo with smaller simulation sizes')
    parser.add_argument('--full-benchmark', action='store_true', 
                       help='Run full performance benchmark')
    parser.add_argument('--stress-test', action='store_true', 
                       help='Run comprehensive stress tests')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("Monte Carlo Risk Management System - Comprehensive Demonstration")
    print("=" * 80)
    print(f"System Configuration:")
    print(f"  CPU Cores: {CONFIG.num_processes}")
    print(f"  Default simulations: {CONFIG.num_simulations:,}")
    print(f"  Time steps: {CONFIG.time_steps}")
    print(f"  Random seed: {CONFIG.random_seed}")
    
    # Adjust configuration for quick mode
    if args.quick:
        CONFIG.num_simulations = 10000
        print(f"\n[QUICK MODE] Reduced simulations to {CONFIG.num_simulations:,}")
    
    results_dict = {}
    benchmark_data = None
    stress_results = None
    
    try:
        # 1. Basic Monte Carlo demonstration
        basic_result = demonstrate_basic_monte_carlo()
        results_dict['basic_gbm'] = basic_result
        
        # 2. Option pricing demonstration
        option_result, analytical_result = demonstrate_option_pricing()
        results_dict['option_pricing'] = option_result
        
        # 3. VaR calculation demonstration
        var_result = demonstrate_var_calculation()
        results_dict['var_calculation'] = var_result
        
        # 4. Parallel processing demonstration
        parallel_results = demonstrate_parallel_processing()
        results_dict.update(parallel_results)
        
        # 5. Performance benchmarking
        if args.full_benchmark:
            print("\nRunning full performance benchmark...")
            benchmark = PerformanceBenchmark()
            benchmark_data = benchmark.run_comprehensive_benchmark()
            benchmark.save_results(benchmark_data, "comprehensive_benchmark")
        else:
            print("\nRunning quick performance benchmark...")
            benchmark_data = run_quick_benchmark()
        
        # 6. Stress testing
        if args.stress_test or not args.quick:
            stress_results = demonstrate_stress_testing()
        
        # 7. Create visualizations
        if not args.no_plots:
            create_visualizations(
                results_dict, 
                benchmark_data, 
                stress_results,
                save_plots=True
            )
        
        # Final summary
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results generated:")
        print(f"  Simulation methods tested: {len(results_dict)}")
        if benchmark_data is not None:
            print(f"  Benchmark test cases: {len(benchmark_data)}")
        if stress_results is not None:
            print(f"  Stress scenarios tested: {len(stress_results)}")
        
        print(f"\nFiles generated:")
        print(f"  monte_carlo_report.html - Comprehensive HTML report")
        if not args.no_plots:
            print(f"  *.png - Visualization plots")
        if benchmark_data is not None:
            print(f"  *benchmark*.csv - Performance benchmark data")
        
        print(f"\nSystem performed {sum(len(r.final_values) for r in results_dict.values() if hasattr(r, 'final_values')):,} total simulations")
        
        # Performance summary
        if 'serial' in results_dict and 'multiprocessing' in results_dict:
            serial_time = results_dict['serial'].execution_time
            parallel_time = results_dict['multiprocessing'].execution_time
            speedup = serial_time / parallel_time
            print(f"Best parallel speedup achieved: {speedup:.2f}x")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()