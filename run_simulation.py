"""
Run Monte Carlo simulation for financial risk management with parallel processing
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
import time

# Import our modules
from financial_models import BlackScholesModel, BlackScholesParameters
from monte_carlo_engine import SimulationResult
from parallel_engine import ParallelMonteCarloEngine

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid")

def visualize_dataset(prices, returns, ticker, output_dir):
    """
    Create visualizations for the dataset processing pipeline
    """
    # Create figure for price history
    fig, ax = plt.subplots(figsize=(12, 6))
    prices.plot(ax=ax)
    ax.set_title(f'{ticker} Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{ticker}_price_history.png'))
    
    # Create figure for returns distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(returns, kde=True, ax=ax)
    mean_return = float(returns.mean())
    ax.axvline(mean_return, color='r', linestyle='--', 
               label=f'Mean: {mean_return:.4f}')
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(f'{ticker} Daily Returns Distribution')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{ticker}_returns_analysis.png'))
    
    return fig

def visualize_performance_comparison(serial_times, parallel_times, num_sims_list, output_dir):
    """
    Create visualization for performance comparison between serial and parallel execution
    """
    # Calculate speedup
    speedup = [s/p for s, p in zip(serial_times, parallel_times)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot execution times
    ax1.plot(num_sims_list, serial_times, 'o-', label='Serial')
    ax1.plot(num_sims_list, parallel_times, 'o-', label='Parallel')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()
    
    # Plot speedup
    ax2.plot(num_sims_list, speedup, 'o-', color='green')
    ax2.set_title('Parallel Processing Speedup (similar to CUDA)')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Speedup Factor (×)')
    ax2.set_xscale('log')
    ax2.grid(True)
    
    fig.suptitle('Performance Analysis of Parallel Monte Carlo Simulations', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    
    return fig

def main():
    # Parameters
    ticker = "AAPL"
    num_simulations = 1000
    horizon_years = 1.0
    
    # Download data
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period="5y")
    
    # Check which price column is available
    if 'Adj Close' in df.columns:
        prices = df['Adj Close']
    elif 'Close' in df.columns:
        prices = df['Close']
    else:
        # Use the first numeric column as price
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                prices = df[col]
                break
    
    # Calculate returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Estimate parameters
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    
    if hasattr(mu, 'item'):
        mu = mu.item()
    if hasattr(sigma, 'item'):
        sigma = sigma.item()
    
    print(f"Asset: {ticker}")
    print(f"Data points: {len(prices)}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Estimated annual drift: {mu:.4f}")
    print(f"Estimated annual volatility: {sigma:.4f}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('reports', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize dataset processing
    visualize_dataset(prices, returns, ticker, output_dir)
    
    # Run Monte Carlo simulation
    days = int(horizon_years * 252)  # Trading days in horizon
    initial_price = float(prices.iloc[-1])
    
    # Create Black-Scholes model
    params = BlackScholesParameters(
        initial_price=initial_price,
        risk_free_rate=mu,
        volatility=sigma
    )
    
    model = BlackScholesModel(params)
    
    # Function to run serial simulation
    def run_serial_simulation(model, num_sims, days, horizon_years):
        start_time = time.time()
        
        # Generate paths
        paths = np.zeros((num_sims, days + 1))
        for i in range(num_sims):
            paths[i] = model.simulate_path(time_horizon=horizon_years, time_steps=days)
        
        # Calculate final values
        final_values = paths[:, -1]
        
        # Create result
        execution_time = time.time() - start_time
        
        return SimulationResult(
            paths=paths,
            final_values=final_values,
            statistics=None,  # Will be calculated in __post_init__
            execution_time=execution_time,
            method="BlackScholes_Serial"
        )
    
    # Function to run parallel simulation
    def run_parallel_simulation(model, num_sims, days, horizon_years, initial_price, mu, sigma):
        from joblib import Parallel, delayed
        import multiprocessing as mp
        
        start_time = time.time()
        
        def simulate_chunk(chunk_id, chunk_size):
            # Create a new model instance for this process
            params = BlackScholesParameters(
                initial_price=initial_price,
                risk_free_rate=mu,
                volatility=sigma
            )
            local_model = BlackScholesModel(params)
            
            # Set different random seed for each chunk
            local_model.random_state = np.random.RandomState(42 + chunk_id)
            
            # Generate paths
            paths = np.zeros((chunk_size, days + 1))
            for i in range(chunk_size):
                paths[i] = local_model.simulate_path(time_horizon=horizon_years, time_steps=days)
            
            return paths
        
        # Determine number of chunks based on CPU cores
        n_jobs = min(mp.cpu_count(), 4)  # Limit to 4 cores max
        chunk_size = num_sims // n_jobs
        if chunk_size == 0:
            chunk_size = 1
            n_jobs = num_sims
        
        # Run simulations in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_chunk)(i, chunk_size if i < n_jobs - 1 else num_sims - i * chunk_size)
            for i in range(n_jobs)
        )
        
        # Combine results
        all_paths = np.vstack(results)
        
        # Create result
        execution_time = time.time() - start_time
        
        return SimulationResult(
            paths=all_paths,
            final_values=all_paths[:, -1],
            statistics=None,  # Will be calculated in __post_init__
            execution_time=execution_time,
            method="BlackScholes_Parallel"
        )
    
    # Run the main simulation
    print(f"Running {num_simulations} Monte Carlo simulations (serial)...")
    serial_result = run_serial_simulation(model, num_simulations, days, horizon_years)
    
    print(f"Running {num_simulations} Monte Carlo simulations (parallel)...")
    parallel_result = run_parallel_simulation(model, num_simulations, days, horizon_years, initial_price, mu, sigma)
    
    # Plot results comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histograms
    sns.histplot(serial_result.final_values, kde=True, ax=ax, 
                 label=f'Serial (time: {serial_result.execution_time:.2f}s)', alpha=0.6)
    sns.histplot(parallel_result.final_values, kde=True, ax=ax, 
                 label=f'Parallel (time: {parallel_result.execution_time:.2f}s)', alpha=0.6)
    
    # Add vertical line for initial price
    ax.axvline(initial_price, color='red', linestyle='--', 
               label=f'Initial Price: ${initial_price:.2f}')
    
    # Calculate VaR
    confidence_level = 0.95
    var_serial = np.percentile(serial_result.final_values, (1 - confidence_level) * 100)
    var_pct_serial = (initial_price - var_serial) / initial_price * 100
    
    # Print summary
    print("\nRisk Metrics:")
    print(f"Initial price: {initial_price:.2f}")
    print(f"VaR ({confidence_level*100:.0f}%): {var_pct_serial:.2f}%")
    print(f"Expected price after {horizon_years} years: {serial_result.statistics['mean']:.2f}")
    print(f"Parallel speedup: {serial_result.execution_time / parallel_result.execution_time:.2f}x")
    
    # Add title and labels
    speedup = serial_result.execution_time / parallel_result.execution_time
    ax.set_title(f'Monte Carlo Results Comparison for {ticker} (Speedup: {speedup:.2f}x)')
    ax.set_xlabel('Final Price')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Save plot
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{ticker}_monte_carlo_comparison.png'))
    
    # Run performance comparison with different simulation sizes
    print("\nRunning performance comparison across different simulation sizes...")
    num_sims_list = [100, 500, 1000, 2000, 5000]
    serial_times = []
    parallel_times = []
    
    for sims in num_sims_list:
        print(f"Testing with {sims} simulations...")
        # Run serial
        sr = run_serial_simulation(model, sims, days, horizon_years)
        serial_times.append(sr.execution_time)
        
        # Run parallel
        pr = run_parallel_simulation(model, sims, days, horizon_years, initial_price, mu, sigma)
        parallel_times.append(pr.execution_time)
    
    # Create performance comparison visualization
    visualize_performance_comparison(serial_times, parallel_times, num_sims_list, output_dir)
    
    # Create accuracy visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot convergence of mean price estimate
    means = []
    for i in range(1, len(serial_result.final_values) + 1, 50):
        means.append(np.mean(serial_result.final_values[:i]))
    
    ax.plot(range(1, len(serial_result.final_values) + 1, 50), means)
    ax.axhline(serial_result.statistics['mean'], color='r', linestyle='--', 
               label=f'Final Mean: {serial_result.statistics["mean"]:.2f}')
    ax.set_title(f'Monte Carlo Simulation Accuracy - Convergence of Mean Estimate')
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Estimated Mean Price')
    ax.legend()
    
    # Save plot
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{ticker}_monte_carlo_accuracy.png'))
    
    print(f"\nResults saved to {output_dir}")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()