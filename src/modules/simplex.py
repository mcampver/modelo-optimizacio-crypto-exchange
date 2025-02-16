import time
from typing import Dict, List
import pandas as pd
import requests
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import scipy.optimize as opt
import praw
from src.modules.simulation import MonteCarloSimulation
import os

class CryptoPortfolioOptimizer:
    def __init__(self, crypto_ids: List[str], initial_investment: float = 10000):
        self.crypto_ids = crypto_ids
        self.initial_investment = initial_investment
        self.historical_data = None
        self.returns = None
        self.optimal_weights = None
        self.expected_return = None
        self.portfolio_risk = None
        self.available_cryptos = []
        self.sentiment_scores = {}
        self.fundamental_data = {}
    
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


    def fetch_sentiment_score(self, crypto_id: str) -> float:
     """
     Obtiene publicaciones relacionadas con la criptomoneda en r/cryptocurrency,
     analiza su sentimiento usando VADER y devuelve el puntaje promedio.
     """
     try:
        # Inicializar el cliente de Reddit usando PRAW
        reddit = praw.Reddit(
            client_id=os.environ.get    ("REDDIT_CLIENT_ID"),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),  # Asegúrate de que este valor sea el correcto
            username=os.environ.get     ("REDDIT_USERNAME"),
            password=os.environ.get     ("REDDIT_PASSWORD"),
            user_agent=os.environ.get   ("REDDIT_USER_AGENT")
        )
        
        # Se busca en el subreddit "cryptocurrency" publicaciones relacionadas con crypto_id.
        subreddit = reddit.subreddit("cryptocurrency")
        posts = subreddit.search(crypto_id, limit=50)
        
        # Inicializar el analizador de sentimiento VADER
        sia = SentimentIntensityAnalyzer()
        compound_scores = []
        
        for post in posts:
            # Combinar el título y el cuerpo de la publicación (selftext) para el análisis
            text = post.title + ". " + (post.selftext or "")
            score = sia.polarity_scores(text)['compound']
            compound_scores.append(score)
        
        # Calcular el promedio de los puntajes compuestos si se encontraron publicaciones
        average_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0.0
        print(f"Puntaje de sentimiento para {crypto_id}: {average_sentiment}")
        return average_sentiment

     except Exception as e:
        print(f"Error al obtener sentimiento para {crypto_id}: {str(e)}")
        return 0.0

    def fetch_fundamental_data(self, crypto_id: str) -> Dict:
        """
        Función para obtener datos fundamentales de la criptomoneda.
        Se utiliza la API de CoinGecko para extraer información como capitalización de mercado, 
        volumen total, etc.
        """
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            fundamental_info = {
                'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd', None),
                'total_volume': data.get('market_data', {}).get('total_volume', {}).get('usd', None),
                'circulating_supply': data.get('market_data', {}).get('circulating_supply', None),
                # Puedes agregar más campos según necesites
            }
            print(f"Datos fundamentales para {crypto_id}: {fundamental_info}")
            return fundamental_info
        except Exception as e:
            print(f"Error obteniendo datos fundamentales para {crypto_id}: {str(e)}")
            return {}


    def integrate_additional_analysis(self):
        """
        Ejecuta el análisis de sentimiento y fundamental para cada criptomoneda disponible,
        almacenando los resultados en los atributos correspondientes.
        """
        for crypto in self.available_cryptos:
            self.sentiment_scores[crypto] = self.fetch_sentiment_score(crypto)
            self.fundamental_data[crypto] = self.fetch_fundamental_data(crypto)
    
    # def optimize_portfolio(self, min_weight: float = 0.0, max_weight: float = 1.0, method: str = 'max_return') -> Dict:
    #  if not self.available_cryptos:
    #     raise ValueError("No hay datos disponibles para optimizar")
    
    #  exp_returns = self.returns.mean() * 252
    
    #  prob = pulp.LpProblem('Portfolio_Optimization', pulp.LpMaximize)
    
    #  weights = pulp.LpVariable.dicts("weights", 
    #                               self.available_cryptos,
    #                               lowBound=min_weight,
    #                               upBound=max_weight)
    
    #  portfolio_return = pulp.lpSum([weights[s] * exp_returns[s] for s in self.available_cryptos])
    #  prob += portfolio_return
    
    #  prob += pulp.lpSum([weights[s] for s in self.available_cryptos]) == 1
    
    #  prob.solve()
    
    #  self.optimal_weights = {s: pulp.value(weights[s]) for s in self.available_cryptos}
    #  self.expected_return = pulp.value(portfolio_return)
    
    #  cov_matrix = self.returns.cov() * 252
    
    # # Verificar si el riesgo se calculó correctamente
    #  if cov_matrix.empty:
    #     self.portfolio_risk = None
    #  else:
    #     self.portfolio_risk = np.sqrt(
    #         sum(self.optimal_weights[i] * self.optimal_weights[j] * cov_matrix.loc[i,j]
    #             for i in self.available_cryptos
    #             for j in self.available_cryptos)
    #     )
    
    #  return {
    #     'weights': self.optimal_weights,
    #     'expected_return': self.expected_return,
    #     'portfolio_risk': self.portfolio_risk,
    #  }

    def optimize_portfolio(self, min_weight: float = 0.0, 
                           max_weight: float = 1.0, method: str = 'max_return') -> Dict:
        if not self.available_cryptos:
            raise ValueError("No hay datos disponibles para optimizar")

        # Cálculo de retornos esperados anuales (asumiendo 252 días hábiles)
        exp_returns = self.returns.mean() * 252

        # Ajuste de los retornos con análisis de sentimiento y datos fundamentales
        adjustment_factor = 0.05  # 5% de ajuste por cada punto de sentimiento
        if not self.sentiment_scores or not self.fundamental_data:
            self.integrate_additional_analysis()

        adjusted_exp_returns = exp_returns.copy()
        for crypto in self.available_cryptos:
            sentiment = self.sentiment_scores.get(crypto, 0)
            # Ajuste según sentimiento
            adjusted_exp_returns[crypto] = exp_returns[crypto] * (1 + adjustment_factor * sentiment)
            # Ajuste según datos fundamentales (por ejemplo, si el market_cap es bajo, se reduce la expectativa)
            fundamental = self.fundamental_data.get(crypto, {})
            market_cap = fundamental.get('market_cap', None)
            if market_cap is not None and market_cap < 1e8:  # Umbral arbitrario
                adjusted_exp_returns[crypto] *= 0.95  # Reducción del 5%

        # Preparar datos para la optimización
        assets = self.available_cryptos
        n = len(assets)
        returns_array = np.array([adjusted_exp_returns[asset] for asset in assets])
        cov_matrix_df = self.returns.cov() * 252
        # Asegurarse de que la matriz de covarianza incluya solo los activos disponibles
        cov_matrix = cov_matrix_df.loc[assets, assets].to_numpy()

        # Restricción: la suma de los pesos debe ser 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        # Límites: cada peso entre min_weight y max_weight
        bounds = [(min_weight, max_weight) for _ in range(n)]
        # Pesos iniciales: asignación equitativa
        initial_weights = np.ones(n) / n

        # Definir la función objetivo según el método seleccionado
        if method == "max_return":
            # Maximizar el retorno esperado => minimizar el negativo del retorno
            objective = lambda w: -np.dot(w, returns_array)
        elif method == "min_variance":
            # Minimizar la varianza del portafolio
            objective = lambda w: np.dot(w, cov_matrix @ w)
        elif method == "max_sharpe":
            # Maximizar el ratio de Sharpe (asumimos tasa libre de riesgo = 0)
            # Se minimiza el negativo del ratio de Sharpe
            # Se agrega una constante pequeña para evitar división por cero.
            objective = lambda w: - (np.dot(w, returns_array)) / np.sqrt(np.dot(w, cov_matrix @ w) + 1e-10)
        else:
            raise ValueError("Método no reconocido. Usa 'max_return', 'min_variance' o 'max_sharpe'.")

        # Resolver la optimización con SLSQP
        result = opt.minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Optimización fallida: " + result.message)

        optimal_weights = result.x
        self.optimal_weights = {asset: weight for asset, weight in zip(assets, optimal_weights)}
        self.expected_return = np.dot(optimal_weights, returns_array)
        self.portfolio_risk = np.sqrt(np.dot(optimal_weights, cov_matrix @ optimal_weights))

        return {
            'weights': self.optimal_weights,
            'expected_return': self.expected_return,
            'portfolio_risk': self.portfolio_risk,
            'sentiment_scores': self.sentiment_scores,
            'fundamental_data': self.fundamental_data,
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
