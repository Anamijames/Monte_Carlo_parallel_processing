"""
Dataset-driven analysis for Monte Carlo risk management using a Kaggle CSV

Usage examples:
  python dataset_analysis.py --csv data/SP500.csv --sims 20000 --horizon 1.0
  python dataset_analysis.py --csv data/AAPL.csv --sims 50000 --horizon 0.5
  python dataset_analysis.py --ticker AAPL --sims 10000 --horizon 0.5

The CSV should contain at least:
  - Date column (e.g., Date)
  - Price column (Adj Close preferred; falls back to Close, then any numeric column)

This script will:
  1) Load the dataset and compute daily returns
  2) Plot price series and returns scatter/histogram
  3) Estimate drift and volatility from the data
  4) Run Monte Carlo (GBM) to show risk measures (e.g., VaR)
  5) Apply parallel processing and compare performance
  6) Save all plots and a summary into reports/<timestamp>/
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import datetime
from tqdm import tqdm
import yfinance as yf
from scipy.stats import norm

# Local modules
from monte_carlo_engine import SimulationResult
from financial_models import BlackScholesParameters, BlackScholesModel
from parallel_engine import ParallelMonteCarloEngine

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

PRICE_CANDIDATES = [
    "Adj Close", "Adjusted Close", "Close", "close", "Adj_Close", "Price", "price"
]


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def load_price_series(csv_path: str = None, ticker: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load price series from a CSV file or download from Yahoo Finance
    
    Args:
        csv_path: Path to CSV file (optional)
        ticker: Stock ticker symbol for Yahoo Finance (optional)
        
    Returns:
        Tuple of (dataframe, price_series)
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        # Find date column
        date_col = None
        for c in df.columns:
            if c.lower() in ("date", "timestamp", "datetime"):
                date_col = c
                break
        
        if date_col is None:
            raise ValueError("Could not find date column in CSV")
        
        # Convert date to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        
        # Find price column
        price_col = None
        for candidate in PRICE_CANDIDATES:
            if candidate in df.columns:
                price_col = candidate
                break
        
        if price_col is None:
            # Fall back to first numeric column
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    price_col = col
                    break
        
        if price_col is None:
            raise ValueError("Could not find price column in CSV")
        
        return df, df[price_col]
    
    elif ticker:
        # Download data from Yahoo Finance
        print(f"Downloading data for {ticker} from Yahoo Finance...")
        df = yf.download(ticker, period="5y")
        # Check which price column is available
        if 'Adj Close' in df.columns:
            price_col = 'Adj Close'
        elif 'Close' in df.columns:
            price_col = 'Close'
        else:
            # Use the first numeric column as price
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    price_col = col
                    break
        return df, df[price_col]
    
    else:
        # Use S&P 500 as default
        print("No CSV or ticker provided. Downloading S&P 500 data...")
        df = yf.download("^GSPC", period="5y")
        return df, df['Adj Close']


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series"""
    return np.log(prices / prices.shift(1)).dropna()


def estimate_parameters(returns: pd.Series, annualization_factor: int = 252) -> Tuple[float, float]:
    """
    Estimate drift and volatility from returns
    
    Args:
        returns: Series of log returns
        annualization_factor: Number of trading days in a year
        
    Returns:
        Tuple of (drift, volatility) annualized
    """
    mu = returns.mean() * annualization_factor
    sigma = returns.std() * np.sqrt(annualization_factor)
    
    # Convert to float if Series
    if hasattr(mu, 'item'):
        mu = mu.item()
    if hasattr(sigma, 'item'):
        sigma = sigma.item()
        
    return mu, sigma


def plot_price_and_returns(df: pd.DataFrame, prices: pd.Series, returns: pd.Series, 
                          output_dir: str, asset_name: str):
    """Create and save plots for price history and returns"""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Price history
    prices.plot(ax=ax1)
    ax1.set_title(f'{asset_name} Price History')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot 2: Returns scatter and histogram
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot of returns
    ax3.scatter(returns.index, returns, alpha=0.5, s=3)
    ax3.set_title(f'{asset_name} Daily Returns')
    ax3.set_ylabel('Log Return')
    ax3.grid(True)
    
    # Histogram of returns with normal distribution overlay
    sns.histplot(returns, kde=True, ax=ax4)
    ax4.set_title(f'{asset_name} Returns Distribution')
    ax4.set_xlabel('Log Return')
    
    # Save plots
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{asset_name}_price_history.png'))
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f'{asset_name}_returns_analysis.png'))
    
    plt.close(fig)
    plt.close(fig2)
    
    return fig, fig2


def run_monte_carlo_simulation(initial_price: float, drift: float, volatility: float, 
                               days: int, num_simulations: int, horizon_years: float) -> SimulationResult:
    """
    Run Monte Carlo simulation using Black-Scholes Model
    
    Args:
        initial_price: Starting price
        drift: Annualized drift (mu)
        volatility: Annualized volatility (sigma)
        days: Number of days to simulate
        num_simulations: Number of simulation paths
        horizon_years: Time horizon in years
        
    Returns:
        SimulationResult object
    """
    # Create Black-Scholes model
    params = BlackScholesParameters(
        initial_price=initial_price,
        risk_free_rate=drift,
        volatility=volatility
    )
    
    model = BlackScholesModel(params)
    
    # Run simulation
    start_time = time.time()
    
    # Generate paths
    paths = np.zeros((num_simulations, days + 1))
    for i in tqdm(range(num_simulations), desc="Simulating paths"):
        paths[i] = model.simulate_path(time_horizon=horizon_years, time_steps=days)
    
    # Calculate final values
    final_values = paths[:, -1]
    
    # Create result
    result = SimulationResult(
        paths=paths,
        final_values=final_values,
        statistics=None,  # Will be calculated in __post_init__
        execution_time=time.time() - start_time,
        method="BlackScholes_Serial"
    )
    
    return result


def run_parallel_monte_carlo(model, num_simulations: int, days: int, 
                            horizon_years: float) -> SimulationResult:
    """Run Monte Carlo simulation in parallel"""
    # Create parallel engine
    parallel_engine = ParallelMonteCarloEngine(model)
    
    # Run simulation
    start_time = time.time()
    
    # Use joblib for parallel processing
    from joblib import Parallel, delayed
    import multiprocessing as mp
    
    def simulate_chunk(chunk_id, chunk_size):
        # Create a new model instance for this process
        params = BlackScholesParameters(
            initial_price=model.parameters.initial_price,
            risk_free_rate=model.parameters.risk_free_rate,
            volatility=model.parameters.volatility
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
    chunk_size = num_simulations // n_jobs
    if chunk_size == 0:
        chunk_size = 1
        n_jobs = num_simulations
    
    # Run simulations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_chunk)(i, chunk_size if i < n_jobs - 1 else num_simulations - i * chunk_size)
        for i in range(n_jobs)
    )
    
    # Combine results
    all_paths = np.vstack(results)
    
    # Create result
    result = SimulationResult(
        paths=all_paths,
        final_values=all_paths[:, -1],
        statistics=None,  # Will be calculated in __post_init__
        execution_time=time.time() - start_time,
        method="BlackScholes_Parallel"
    )
    
    return result


def plot_monte_carlo_results(result: SimulationResult, output_dir: str, 
                            asset_name: str, confidence_level: float = 0.95):
    """Plot Monte Carlo simulation results"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot a subset of paths
    num_paths_to_plot = min(100, result.paths.shape[0])
    for i in range(num_paths_to_plot):
        ax.plot(result.paths[i], alpha=0.1, color='blue')
    
    # Plot mean path
    mean_path = np.mean(result.paths, axis=0)
    ax.plot(mean_path, color='red', linewidth=2, label='Mean Path')
    
    # Calculate VaR
    var_level = np.percentile(result.final_values, (1 - confidence_level) * 100)
    initial_price = result.paths[0, 0]
    var_pct = (initial_price - var_level) / initial_price * 100
    
    # Add VaR line
    ax.axhline(y=var_level, color='red', linestyle='--', 
              label=f'VaR ({confidence_level*100:.0f}%): {var_pct:.2f}%')
    
    ax.set_title(f'Monte Carlo Simulation for {asset_name} ({result.method})')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{asset_name}_monte_carlo_{result.method}.png'))
    
    plt.close(fig)
    
    return fig


def plot_comparison(serial_result: SimulationResult, parallel_result: SimulationResult, 
                   output_dir: str, asset_name: str):
    """Plot comparison of serial and parallel Monte Carlo results"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histograms
    sns.histplot(serial_result.final_values, kde=True, alpha=0.5, 
                label=f'Serial ({serial_result.execution_time:.2f}s)', ax=ax)
    sns.histplot(parallel_result.final_values, kde=True, alpha=0.5, 
                label=f'Parallel ({parallel_result.execution_time:.2f}s)', ax=ax)
    
    # Calculate speedup
    speedup = serial_result.execution_time / parallel_result.execution_time
    
    ax.set_title(f'Monte Carlo Results Comparison for {asset_name} (Speedup: {speedup:.2f}x)')
    ax.set_xlabel('Final Price')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Save plot
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{asset_name}_monte_carlo_comparison.png'))
    
    plt.close(fig)
    
    return fig


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monte Carlo Risk Analysis')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--sims', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--horizon', type=float, default=1.0, help='Time horizon in years')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('reports', timestamp)
    ensure_dir(output_dir)
    
    # Load data
    df, prices = load_price_series(args.csv, args.ticker)
    
    # Determine asset name
    if args.ticker:
        asset_name = args.ticker
    elif args.csv:
        asset_name = os.path.basename(args.csv).split('.')[0]
    else:
        asset_name = 'SP500'
    
    # Calculate returns and parameters
    returns = calculate_returns(prices)
    drift, volatility = estimate_parameters(returns)
    
    print(f"Asset: {asset_name}")
    print(f"Data points: {len(prices)}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Estimated annual drift: {drift:.4f}")
    print(f"Estimated annual volatility: {volatility:.4f}")
    
    # Plot price and returns
    plot_price_and_returns(df, prices, returns, output_dir, asset_name)
    
    # Run Monte Carlo simulation
    days = int(args.horizon * 252)  # Trading days in horizon
    initial_price = float(prices.iloc[-1])
    
    # Create Black-Scholes model
    params = BlackScholesParameters(
        initial_price=initial_price,
        risk_free_rate=drift,
        volatility=volatility
    )
    
    model = BlackScholesModel(params)
    
    # Run serial simulation
    print(f"Running {args.sims} Monte Carlo simulations (serial)...")
    serial_result = run_monte_carlo_simulation(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        num_simulations=args.sims,
        days=days,
        horizon_years=args.horizon
    )
    
    # Plot serial results
    plot_monte_carlo_results(serial_result, output_dir, asset_name)
    
    # Run parallel simulation
    print(f"Running {args.sims} Monte Carlo simulations (parallel)...")
    parallel_result = run_parallel_monte_carlo(
        model=model,
        num_simulations=args.sims,
        days=days,
        horizon_years=args.horizon
    )
    
    # Plot parallel results
    plot_monte_carlo_results(parallel_result, output_dir, asset_name)
    
    # Plot comparison
    plot_comparison(serial_result, parallel_result, output_dir, asset_name)
    
    # Calculate VaR and other risk metrics
    confidence_level = 0.95
    var_serial = np.percentile(serial_result.final_values, (1 - confidence_level) * 100)
    var_pct_serial = (initial_price - var_serial) / initial_price * 100
    
    var_parallel = np.percentile(parallel_result.final_values, (1 - confidence_level) * 100)
    var_pct_parallel = (initial_price - var_parallel) / initial_price * 100
    
    # Print summary
    print("\nRisk Metrics:")
    print(f"Initial price: {initial_price:.2f}")
    print(f"VaR ({confidence_level*100:.0f}%): {var_pct_serial:.2f}%")
    print(f"Expected price after {args.horizon} years: {serial_result.statistics['mean']:.2f}")
    print(f"Parallel speedup: {serial_result.execution_time / parallel_result.execution_time:.2f}x")
    
    # Save summary to file
    with open(os.path.join(output_dir, f'{asset_name}_summary.txt'), 'w') as f:
        f.write(f"Asset: {asset_name}\n")
        f.write(f"Data points: {len(prices)}\n")
        f.write(f"Date range: {prices.index[0]} to {prices.index[-1]}\n")
        f.write(f"Estimated annual drift: {drift:.4f}\n")
        f.write(f"Estimated annual volatility: {volatility:.4f}\n\n")
        
        f.write("Risk Metrics:\n")
        f.write(f"Initial price: {initial_price:.2f}\n")
        f.write(f"VaR ({confidence_level*100:.0f}%): {var_pct_serial:.2f}%\n")
        f.write(f"Expected price after {args.horizon} years: {serial_result.statistics['mean']:.2f}\n")
        f.write(f"Parallel speedup: {serial_result.execution_time / parallel_result.execution_time:.2f}x\n")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()


def plot_dataset(df: pd.DataFrame, returns: pd.Series, outdir: str):
    ensure_dir(outdir)
    # 1) Price series
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Price"], color="#1f77b4")
    plt.title("Price Series")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "price_series.png"), dpi=150)
    plt.close()

    # 2) Returns histogram + KDE
    plt.figure(figsize=(10, 5))
    sns.histplot(returns, bins=50, kde=True, color="#2ca02c")
    plt.title("Daily Returns Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "returns_hist.png"), dpi=150)
    plt.close()

    # 3) Returns scatter vs lagged (autocorrelation look)
    lagged = returns.shift(1).dropna()
    aligned = returns.loc[lagged.index]
    plt.figure(figsize=(6, 6))
    plt.scatter(lagged, aligned, alpha=0.3, s=10, color="#ff7f0e")
    plt.title("Returns vs Lagged Returns")
    plt.xlabel("Return (t-1)")
    plt.ylabel("Return (t)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "returns_scatter.png"), dpi=150)
    plt.close()


def run_monte_carlo(initial_price: float, mu_annual: float, sigma_annual: float, sims: int, horizon: float) -> np.ndarray:
    engine = GeometricBrownianMotionEngine(
        initial_price=initial_price,
        drift=mu_annual,
        volatility=sigma_annual,
        time_horizon=horizon,
    )
    # Vectorized for speed & reproducibility
    result = engine.simulate_vectorized(sims)
    return result.final_values


def calculate_var(pnl: np.ndarray, level: float = 0.95) -> float:
    # pnl distribution: negative values are losses; VaR is a positive number for the loss
    return -np.percentile(pnl, (1 - level) * 100)


def plot_mc_results(final_prices: np.ndarray, initial_price: float, outdir: str, label: str):
    ensure_dir(outdir)
    # Convert to P&L terms for interpretation
    pnl = final_prices - initial_price
    var95 = calculate_var(pnl, 0.95)
    var99 = calculate_var(pnl, 0.99)

    plt.figure(figsize=(10, 5))
    sns.histplot(pnl, bins=60, kde=True, color="#6a5acd")
    plt.axvline(-var95, color="red", linestyle="--", label=f"VaR 95% = {var95:,.0f}")
    plt.axvline(-var99, color="darkred", linestyle=":", label=f"VaR 99% = {var99:,.0f}")
    plt.title(f"Monte Carlo P&L Distribution ({label})")
    plt.xlabel("P&L")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"mc_pnl_{label}.png"), dpi=150)
    plt.close()

    return var95, var99


def compare_parallel(initial_price: float, mu_annual: float, sigma_annual: float, sims: int, horizon: float, outdir: str):
    ensure_dir(outdir)
    engine = GeometricBrownianMotionEngine(
        initial_price=initial_price,
        drift=mu_annual,
        volatility=sigma_annual,
        time_horizon=horizon,
    )
    parallel_engine = ParallelMonteCarloEngine(engine)

    # Measure three methods: serial, vectorized, multiprocessing
    timings = {}
    finals = {}

    t0 = time.time()
    serial_res = engine.simulate_serial(sims)
    timings["serial"] = time.time() - t0
    finals["serial"] = serial_res.final_values

    t0 = time.time()
    vec_res = engine.simulate_vectorized(sims)
    timings["vectorized"] = time.time() - t0
    finals["vectorized"] = vec_res.final_values

    t0 = time.time()
    mp_res = parallel_engine.simulate_multiprocessing(sims)
    timings["multiprocessing"] = time.time() - t0
    finals["multiprocessing"] = mp_res.final_values

    # Bar chart for timings
    plt.figure(figsize=(8, 4))
    methods = list(timings.keys())
    values = [timings[m] for m in methods]
    sns.barplot(x=methods, y=values, palette="Blues_d")
    plt.title(f"Execution Time (sims={sims:,})")
    plt.ylabel("Seconds")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "parallel_timings.png"), dpi=150)
    plt.close()

    # Overlay P&L distributions (should be statistically similar)
    plt.figure(figsize=(10, 5))
    for m, vals in finals.items():
        pnl = vals - initial_price
        sns.kdeplot(pnl, label=m)
    plt.title("P&L KDE by Method")
    plt.xlabel("P&L")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "parallel_kde.png"), dpi=150)
    plt.close()

    return timings


def main():
    parser = argparse.ArgumentParser(description="Kaggle dataset analysis + Monte Carlo risk + parallel processing")
    parser.add_argument("--csv", required=True, help="Path to Kaggle CSV (must include a Date and Price/Close column)")
    parser.add_argument("--sims", type=int, default=20000, help="Number of Monte Carlo simulations")
    parser.add_argument("--horizon", type=float, default=1.0, help="Time horizon in years (e.g., 1.0 = one year)")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("reports", f"kaggle_analysis_{ts}")
    ensure_dir(outdir)

    print(f"Loading dataset from: {args.csv}")
    df, price = load_price_series(args.csv)
    returns = compute_returns(price)

    print("Generating base plots ...")
    plot_dataset(df, returns, outdir)

    print("Estimating drift/volatility from historical returns ...")
    mu_annual, sigma_annual = estimate_annual_mu_sigma(returns)
    initial_price = float(price.iloc[-1])
    print(f"Initial price: {initial_price:.2f}")
    print(f"Estimated annual drift: {mu_annual:.4f}")
    print(f"Estimated annual volatility: {sigma_annual:.4f}")

    print("Running Monte Carlo (vectorized) ...")
    final_prices = run_monte_carlo(initial_price, mu_annual, sigma_annual, args.sims, args.horizon)
    var95, var99 = plot_mc_results(final_prices, initial_price, outdir, label="vectorized")
    print(f"VaR 95% (1-year): {var95:,.0f}")
    print(f"VaR 99% (1-year): {var99:,.0f}")

    print("Comparing parallel processing methods ...")
    timings = compare_parallel(initial_price, mu_annual, sigma_annual, args.sims, args.horizon, outdir)
    for m, tsec in timings.items():
        print(f"  {m:15s}: {tsec:.3f} s")

    print("All plots saved in:", outdir)


if __name__ == "__main__":
    main()
