"""
Visualization and reporting tools for Monte Carlo simulations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats

from monte_carlo_engine import SimulationResult
from config import CONFIG

# Set plotting styles
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
sns.set_palette("husl")

class MonteCarloVisualizer:
    """Comprehensive visualization toolkit for Monte Carlo results"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.figure_size = (12, 8)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_simulation_paths(self, result: SimulationResult, 
                             num_paths_to_plot: int = 100,
                             title: str = "Monte Carlo Simulation Paths") -> plt.Figure:
        """Plot sample simulation paths"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Select random subset of paths to plot
        num_paths = min(num_paths_to_plot, len(result.paths))
        selected_indices = np.random.choice(len(result.paths), num_paths, replace=False)
        selected_paths = result.paths[selected_indices]
        
        # Time axis
        time_axis = np.linspace(0, 1, selected_paths.shape[1])  # Assuming 1 year horizon
        
        # Plot paths with transparency
        for path in selected_paths:
            ax.plot(time_axis, path, alpha=0.3, linewidth=0.8, color='blue')
        
        # Plot mean path
        mean_path = np.mean(result.paths, axis=0)
        ax.plot(time_axis, mean_path, color='red', linewidth=2, 
                label=f'Mean Path (Final: {mean_path[-1]:.2f})')
        
        # Confidence bands
        percentile_5 = np.percentile(result.paths, 5, axis=0)
        percentile_95 = np.percentile(result.paths, 95, axis=0)
        ax.fill_between(time_axis, percentile_5, percentile_95, 
                       alpha=0.2, color='gray', label='90% Confidence Band')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Asset Price')
        ax.set_title(f'{title}\\n{len(result.paths):,} simulations ({result.method})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_final_value_distribution(self, result: SimulationResult,
                                     bins: int = 50,
                                     title: str = "Final Value Distribution") -> plt.Figure:
        """Plot distribution of final values"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(result.final_values, bins=bins, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        
        # Fit normal distribution for comparison
        mu, sigma = stats.norm.fit(result.final_values)
        x = np.linspace(result.final_values.min(), result.final_values.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
        
        ax1.set_xlabel('Final Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{title} - Histogram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(result.final_values, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot vs Normal Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(result.final_values):.2f}\\n'
        stats_text += f'Std: {np.std(result.final_values):.2f}\\n'
        stats_text += f'Skewness: {stats.skew(result.final_values):.3f}\\n'
        stats_text += f'Kurtosis: {stats.kurtosis(result.final_values):.3f}\\n'
        stats_text += f'Min: {np.min(result.final_values):.2f}\\n'
        stats_text += f'Max: {np.max(result.final_values):.2f}'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_var_analysis(self, result: SimulationResult,
                         confidence_levels: List[float] = None,
                         title: str = "Value at Risk Analysis") -> plt.Figure:
        """Plot VaR analysis with multiple confidence levels"""
        
        if confidence_levels is None:
            confidence_levels = self.config.confidence_levels
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sort values for VaR calculation
        sorted_values = np.sort(result.final_values)
        
        # Plot 1: Distribution with VaR lines
        ax1.hist(result.final_values, bins=50, density=True, alpha=0.7, 
                color='lightblue', edgecolor='black')
        
        colors_var = ['red', 'orange', 'purple']
        var_values = []
        
        for i, confidence_level in enumerate(confidence_levels):
            percentile = (1 - confidence_level) * 100
            var_value = np.percentile(sorted_values, percentile)
            var_values.append(var_value)
            
            ax1.axvline(var_value, color=colors_var[i % len(colors_var)], 
                       linestyle='--', linewidth=2,
                       label=f'VaR {confidence_level:.1%}: {var_value:.2f}')
        
        ax1.set_xlabel('Portfolio Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{title} - Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: VaR vs Confidence Level
        confidence_range = np.linspace(0.90, 0.999, 50)
        var_curve = [np.percentile(sorted_values, (1-cl)*100) for cl in confidence_range]
        
        ax2.plot(confidence_range * 100, var_curve, 'b-', linewidth=2)
        
        # Mark the specific confidence levels
        for i, (cl, var_val) in enumerate(zip(confidence_levels, var_values)):
            ax2.plot(cl * 100, var_val, 'o', color=colors_var[i % len(colors_var)], 
                    markersize=8, label=f'{cl:.1%}: {var_val:.2f}')
        
        ax2.set_xlabel('Confidence Level (%)')
        ax2.set_ylabel('Value at Risk')
        ax2.set_title('VaR vs Confidence Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self, results: Dict[str, SimulationResult],
                                    benchmark_data: Optional[pd.DataFrame] = None,
                                    save_path: Optional[str] = None) -> str:
        """Generate comprehensive HTML report"""
        
        html_content = []
        html_content.append("<html><head><title>Monte Carlo Risk Analysis Report</title>")
        html_content.append("<style>")
        html_content.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        html_content.append("h1, h2 { color: #2E86AB; }")
        html_content.append("table { border-collapse: collapse; width: 100%; }")
        html_content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_content.append("th { background-color: #f2f2f2; }")
        html_content.append("</style></head><body>")
        
        # Title and summary
        html_content.append("<h1>Monte Carlo Risk Analysis Report</h1>")
        html_content.append(f"<p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Simulation results summary
        html_content.append("<h2>Simulation Results Summary</h2>")
        html_content.append("<table>")
        html_content.append("<tr><th>Method</th><th>Simulations</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Execution Time</th></tr>")
        
        for method, result in results.items():
            html_content.append(f"<tr>")
            html_content.append(f"<td>{method}</td>")
            html_content.append(f"<td>{len(result.final_values):,}</td>")
            html_content.append(f"<td>{np.mean(result.final_values):.2f}</td>")
            html_content.append(f"<td>{np.std(result.final_values):.2f}</td>")
            html_content.append(f"<td>{np.min(result.final_values):.2f}</td>")
            html_content.append(f"<td>{np.max(result.final_values):.2f}</td>")
            html_content.append(f"<td>{result.execution_time:.3f}s</td>")
            html_content.append(f"</tr>")
        
        html_content.append("</table>")
        
        # Risk metrics
        html_content.append("<h2>Risk Metrics</h2>")
        for method, result in results.items():
            if hasattr(result, 'statistics') and result.statistics:
                html_content.append(f"<h3>{method}</h3>")
                html_content.append("<table>")
                
                for key, value in result.statistics.items():
                    if isinstance(value, (int, float)):
                        html_content.append(f"<tr><td>{key}</td><td>{value:.4f}</td></tr>")
                    else:
                        html_content.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                
                html_content.append("</table>")
        
        # Performance benchmark results
        if benchmark_data is not None and len(benchmark_data) > 0:
            html_content.append("<h2>Performance Benchmark Results</h2>")
            html_content.append(benchmark_data.to_html(index=False, table_id="benchmark_table"))
        
        html_content.append("</body></html>")
        
        html_report = "\\n".join(html_content)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html_report)
            print(f"Report saved to: {save_path}")
        
        return html_report

def plot_stress_test_results(stress_results: Dict[str, Any], 
                           title: str = "Stress Test Results") -> plt.Figure:
    """Plot stress test results comparison"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    scenarios = list(stress_results.keys())
    
    # Extract VaR values for different confidence levels
    var_95 = []
    var_99 = []
    mean_pnl = []
    max_loss = []
    
    for scenario in scenarios:
        result = stress_results[scenario]['simulation_result']
        stats = result.statistics
        
        var_95.append(stats.get('VaR_0.95', 0))
        var_99.append(stats.get('VaR_0.99', 0))
        mean_pnl.append(stats.get('mean_pnl', 0))
        max_loss.append(stats.get('min_pnl', 0))
    
    # Plot 1: VaR comparison
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x - width/2, var_95, width, label='VaR 95%', alpha=0.8)
    ax1.bar(x + width/2, var_99, width, label='VaR 99%', alpha=0.8)
    ax1.set_xlabel('Stress Scenarios')
    ax1.set_ylabel('Value at Risk')
    ax1.set_title('VaR under Different Stress Scenarios')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean P&L
    ax2.bar(scenarios, mean_pnl, color='lightblue', alpha=0.8)
    ax2.set_xlabel('Stress Scenarios')
    ax2.set_ylabel('Mean P&L')
    ax2.set_title('Mean P&L under Stress Scenarios')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Maximum Loss
    ax3.bar(scenarios, max_loss, color='red', alpha=0.8)
    ax3.set_xlabel('Stress Scenarios')
    ax3.set_ylabel('Maximum Loss')
    ax3.set_title('Worst Case Loss by Scenario')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: P&L distributions for each scenario
    for i, scenario in enumerate(scenarios):
        result = stress_results[scenario]['simulation_result']
        pnl_values = result.final_values
        
        ax4.hist(pnl_values, bins=30, alpha=0.6, label=scenario, density=True)
    
    ax4.set_xlabel('P&L')
    ax4.set_ylabel('Density')
    ax4.set_title('P&L Distributions by Scenario')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig