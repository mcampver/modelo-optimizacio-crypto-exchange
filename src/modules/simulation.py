import numpy as np
import plotly.graph_objects as go

class MonteCarloSimulation:
    def __init__(self, returns: np.ndarray, num_simulations: int = 1000, days: int = 252):
        self.returns = returns
        self.num_simulations = num_simulations
        self.days = days
        self.simulated_portfolios = None
    
    def run_simulation(self, weights: np.ndarray):
        mean_returns = np.mean(self.returns, axis=0)
        cov_matrix = np.cov(self.returns.T)
        
        portfolio_simulations = np.zeros((self.num_simulations, self.days))
        
        for i in range(self.num_simulations):
            daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, self.days)
            portfolio_returns = np.cumprod(1 + np.dot(daily_returns, weights))
            portfolio_simulations[i, :] = portfolio_returns
        
        self.simulated_portfolios = portfolio_simulations
        return portfolio_simulations
    
    def plot_simulation(self):
        if self.simulated_portfolios is None:
            raise ValueError("Debe ejecutar run_simulation primero.")
        
        fig = go.Figure()

        # Añadir cada simulación como una traza (trace) en el gráfico
        for i in range(self.num_simulations):
            fig.add_trace(go.Scatter(x=np.arange(self.days), y=self.simulated_portfolios[i], 
                                     mode='lines', line=dict(color='blue', width=1, opacity=0.1)))

        fig.update_layout(
            title="Simulación Monte Carlo de Portafolio",
            xaxis_title="Días",
            yaxis_title="Valor del Portafolio (Normalizado)",
            template="plotly_dark"
        )

        fig.show()
    
    def get_statistics(self):
        if self.simulated_portfolios is None:
            raise ValueError("Debe ejecutar run_simulation primero.")
        
        final_values = self.simulated_portfolios[:, -1]
        percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
        
        return {
            "percentile_5": percentiles[0],
            "percentile_25": percentiles[1],
            "median": percentiles[2],
            "percentile_75": percentiles[3],
            "percentile_95": percentiles[4],
            "mean": np.mean(final_values),
            "std_dev": np.std(final_values)
        }
