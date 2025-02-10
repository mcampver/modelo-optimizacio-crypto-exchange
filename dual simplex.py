import numpy as np
import pandas as pd
import pulp
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import List, Dict
import time

class CryptoPortfolioOptimizer:
    def __init__(self, crypto_ids: List[str], initial_investment: float = 10000, liquidity_data: Dict[str, float] = None):

      self.crypto_ids = crypto_ids
      self.initial_investment = initial_investment
      self.historical_data = None
      self.returns = None
      self.optimal_weights = None
      self.expected_return = None
      self.portfolio_risk = None
      self.liquidity_data = liquidity_data if liquidity_data else {}  # Diccionario de liquidez por criptomoneda
      self.available_cryptos = []
    
    def fetch_historical_data(self, days: int = 365, retry_delay: int = 60) -> pd.DataFrame:
        data = pd.DataFrame()
        
        for crypto_id in self.crypto_ids:
            success = False
            retries = 3
            
            while not success and retries > 0:
                try:
                    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
                    params = {'vs_currency': 'usd', 'days': str(days), 'interval': 'daily'}
                    
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 429:
                        print(f"Límite de API alcanzado. Esperando {retry_delay} segundos...")
                        time.sleep(retry_delay)
                        retries -= 1
                        continue
                    
                    response.raise_for_status()
                    prices = response.json()['prices']
                    
                    df = pd.DataFrame(prices, columns=['timestamp', crypto_id])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    data[crypto_id] = df[crypto_id]
                    self.available_cryptos.append(crypto_id)
                    success = True
                    print(f"Datos obtenidos exitosamente para {crypto_id}")
                    
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error obteniendo datos para {crypto_id}: {str(e)}")
                    retries -= 1
                    if retries > 0:
                        print(f"Reintentando... ({retries} intentos restantes)")
                        time.sleep(5)
            
            if not success:
                print(f"No se pudieron obtener datos para {crypto_id} después de varios intentos")
        
        if len(self.available_cryptos) < 2:
            raise ValueError("No hay suficientes criptomonedas con datos disponibles para optimizar el portafolio")
        
        self.historical_data = data
        self.returns = data.pct_change(fill_method=None).dropna()

        return self.historical_data

    def optimize_portfolio(self, min_weight: float = 0.0, max_weight: float = 1.0, min_capital: float = 1000, max_capital: float = 5000, max_risk: float = 0.1) -> Dict:
      if not self.available_cryptos:
        raise ValueError("No hay datos disponibles para optimizar")
    
    # Calcular los rendimientos esperados anuales
      exp_returns = self.returns.mean() * 252
      print(exp_returns)
    # Crear el modelo de optimización
      prob = pulp.LpProblem('Portfolio_Optimization', pulp.LpMaximize)
    
    # Crear las variables de peso para cada criptomoneda
      weights = pulp.LpVariable.dicts("weights", self.available_cryptos, lowBound=min_weight, upBound=max_weight)
    
    # Función objetivo: maximizar el retorno esperado del portafolio
      portfolio_return = pulp.lpSum([weights[s] * exp_returns[s] for s in self.available_cryptos])
      prob += portfolio_return
    
    # Restricción de que los pesos sumen 1
      prob += pulp.lpSum([weights[s] for s in self.available_cryptos]) == 1
    
    # Restricción de capital mínimo y máximo por activo
      for crypto in self.available_cryptos:
        prob += weights[crypto] * self.initial_investment >= min_capital  # Capital mínimo por activo
        prob += weights[crypto] * self.initial_investment <= max_capital  # Capital máximo por activo
    
    # Restricción de liquidez mínima
      if self.liquidity_data:
        for crypto in self.available_cryptos:
            if crypto in self.liquidity_data:
                liquidity_threshold = self.liquidity_data[crypto]
                prob += weights[crypto] * self.initial_investment >= liquidity_threshold  # Excluir criptos con baja liquidez
    
    # Solucionar el modelo sin la restricción de riesgo aún
      prob.solve()

    # Obtener los pesos óptimos
      self.optimal_weights = {s: pulp.value(weights[s]) for s in self.available_cryptos}
    
    # Calcular el riesgo fuera del modelo de optimización
      cov_matrix = self.returns.cov() * 252
      portfolio_risk = np.sqrt(
        sum(self.optimal_weights[i] * self.optimal_weights[j] * cov_matrix.loc[i,j] for i in self.available_cryptos for j in self.available_cryptos)
      )
    
    # Verificar si el riesgo excede el límite
      if portfolio_risk > max_risk:
        print(f"El riesgo total del portafolio ({portfolio_risk}) excede el límite permitido de {max_risk}. Ajustando la optimización.")
    
      return {
        'weights': self.optimal_weights,
        'expected_return': pulp.value(portfolio_return),
        'portfolio_risk': portfolio_risk,
      }

    def get_invested_values(self):
     if self.optimal_weights is None:
        raise ValueError("Debe ejecutar optimize_portfolio primero")

     invested_values = {crypto: self.optimal_weights[crypto] * self.initial_investment for crypto in self.available_cryptos}
     return invested_values


    
    def run_monte_carlo_simulation(self, num_simulations: int = 1000, days: int = 252):
        if self.optimal_weights is None:
            raise ValueError("Debe ejecutar optimize_portfolio primero")
        
        weights = np.array([self.optimal_weights[crypto] for crypto in self.available_cryptos])
        monte_carlo = MonteCarloSimulation(self.returns.to_numpy(), num_simulations, days)
        simulations = monte_carlo.run_simulation(weights)
        
        monte_carlo.plot_simulation()
        stats = monte_carlo.get_statistics()
        print("Resultados de Monte Carlo:", stats)
        return stats

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
                                     mode='lines', line=dict(color='blue', width=1), opacity=0.1))

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
        

        
def main():
    crypto_ids = ['bitcoin', 'ethereum', 'ripple', 'tether', 'solana', 'binancecoin', 'usd-coin', 'dogecoin', 'cardano', 'staked-ether']
    investment = 10000
    
    optimizer = CryptoPortfolioOptimizer(crypto_ids, investment)
    
    try:
        optimizer.fetch_historical_data(days=365, retry_delay=60)
        optimizer.optimize_portfolio(min_weight=0.05, max_weight=0.4)
        
        invested_values = optimizer.get_invested_values()
        print("Valores invertidos en cada criptomoneda:")
        for crypto, value in invested_values.items():
            print(f"{crypto}: ${value:.2f}")
        # Ejecutar Monte Carlo
        # monte_carlo_results = optimizer.run_monte_carlo_simulation(num_simulations=1000, days=252)
        
        # print(monte_carlo_results)
        
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()
