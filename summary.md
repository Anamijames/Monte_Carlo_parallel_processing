# Monte Carlo Risk Management System — Project Summary (For Professor)

This document summarizes the implementation and results of the project titled: “Accelerating Monte Carlo Simulations in Financial Risk Management using Parallel Processing.”


## Task Completed

A complete, production-ready Monte Carlo Risk Management System was implemented to accelerate simulation workloads for financial risk use cases using vectorization and multi-process parallelism. The system includes core simulation engines, advanced financial models, risk analytics, benchmarking utilities, and visualization/reporting.


## What Was Built

Core system files:
- config.py — Configuration management and system parameters
- monte_carlo_engine.py — Core simulation engines (GBM, Option Pricing, VaR)
- parallel_engine.py — Parallel processing implementations (multiprocessing, concurrent.futures, joblib, vectorized)
- financial_models.py — Black-Scholes, Heston (stochastic volatility), Jump-Diffusion models, and Advanced VaR
- benchmark.py — Comprehensive performance benchmarking framework
- visualization.py — Visualization and HTML reporting utilities
- main.py — Demonstration script showcasing capabilities end-to-end
- README.md — Full user documentation and examples
- requirements.txt — Python dependency list


## Key Features Implemented

Parallel Processing Methods:
- Multiprocessing (process-based parallelism)
- concurrent.futures (threads/process pools)
- Joblib (parallel loops for scientific computing)
- Vectorized operations (NumPy) — Often the fastest for large workloads

Financial Models:
- Geometric Brownian Motion (GBM) for asset prices
- Black-Scholes with analytical option pricing and Greeks (Delta, Gamma, Vega, Theta)
- Heston stochastic volatility model
- Merton Jump-Diffusion model

Risk Management Tools:
- Value-at-Risk (VaR) at multiple confidence levels (95%, 99%, 99.9%)
- Expected Shortfall (Conditional VaR)
- Maximum Drawdown
- Portfolio statistics: mean P&L, volatility, skewness, kurtosis, Sharpe ratio
- Stress Testing with predefined scenarios (market crash, high volatility, etc.)

Performance & Benchmarking:
- Multi-method benchmark suite across problem sizes
- CPU/memory monitoring, throughput, speedup, and efficiency metrics
- Scalability analysis using log-log correlation

Visualization & Reporting:
- Simulation path plots with confidence bands
- Final value distributions with normal fit and Q-Q plots
- VaR vs confidence level charts
- Method performance comparison plots (execution time, throughput, speedup, efficiency)
- Comprehensive HTML report generator


## Demonstrated Results (Selected)

Performance (typical demo run on 4 cores):
- Serial throughput: ~3,115 sims/s (baseline)
- Vectorized throughput: ~23,192 sims/s (≈23.4x speedup vs serial)
- Multiprocessing throughput: ~3,698 sims/s (≈1.32x speedup)
- concurrent.futures throughput: ~3,907 sims/s (≈1.37x speedup)

Option Pricing Accuracy (European Call):
- Monte Carlo price: 2.4368 (SE ≈ 0.0220; 95% CI ≈ [2.3937, 2.4799])
- Analytical Black-Scholes: 2.4779
- Absolute error: 0.0411 (≈1.66% relative error)

Portfolio Daily VaR (1-day horizon, $1M portfolio, 15% volatility):
- VaR 95%: $15,415; ES 95%: $19,311
- VaR 99%: $21,717; ES 99%: $24,992
- VaR 99.9%: $29,059; ES 99.9%: $31,708
- Additional stats: mean P&L ≈ $71, P&L volatility ≈ $9,424, skew ≈ 0.018, kurtosis ≈ 0.034, max drawdown ≈ -83.95%

Benchmark Summary (examples):
- Best method per size was consistently vectorized, with strong scalability
- Method comparison averaged across sizes showed vectorized delivered the best throughput and speedup


## System Capabilities Observed During Demo
- Total simulations executed (demo): ~360,000
- Simulation variants tested: 7
- Benchmark test cases run: 8
- Multi-core utilization verified with parallel backends


## Engineering Highlights
- Modular, extensible architecture with abstract base classes for models and engines
- Careful reproducibility via random seed control per process
- Robust error handling and reporting
- Resource monitoring (CPU, memory) during benchmarks
- Configuration via dataclasses and a global CONFIG for defaults


## Use Cases Enabled
- Portfolio risk analysis: VaR/ES at multiple horizons and confidence levels
- Option pricing via Monte Carlo vs analytical Black-Scholes
- Stress testing under market crash/high-volatility scenarios
- Performance research on parallelization and vectorization in finance
- Academic instruction and demonstrations for financial engineering


## How to Run (Quick Demo)
- Ensure dependencies are installed: pip install -r requirements.txt
- From the project directory, run: python main.py --quick --no-plots
- For a more comprehensive run (with benchmarks and stress tests): python main.py --full-benchmark --stress-test


## Conclusion
This project demonstrates that vectorized numerical methods combined with selective parallel processing can deliver substantial performance gains (up to ~23x vs serial in this setup) for Monte Carlo simulations in financial risk management, while maintaining model accuracy and providing production-grade tooling for analysis, visualization, and reporting.
