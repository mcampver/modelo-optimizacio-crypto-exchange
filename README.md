# modelo-optimizacio-crypto-exchange

## Introducción
Este proyecto es un modelo de optimización basado en Python para un intercambio de criptomonedas. Utiliza métodos avanzados de optimización de portafolios e integra análisis de sentimiento y datos fundamentales para mejorar las decisiones de inversión.

## Componente Principal: `simplex.py`

### Descripción
El archivo `simplex.py` es el núcleo del proyecto y se encarga de optimizar portafolios de criptomonedas mediante diferentes métodos como la maximización de retorno, mínima varianza, y maximización del ratio de Sharpe. Además, integra análisis de sentimiento y datos fundamentales para ajustar las expectativas de retorno.

### Funcionalidades:
- **Obtención de Datos Históricos**: Recopila datos históricos de precios de criptomonedas desde la API de CoinGecko.
- **Análisis de Sentimiento**: Analiza el sentimiento de publicaciones de Reddit relacionadas con las criptomonedas utilizando VADER.
- **Datos Fundamentales**: Obtiene datos fundamentales de las criptomonedas desde la API de CoinGecko.
- **Optimización de Portafolio**: Optimiza el portafolio basado en diferentes métodos ajustando los retornos esperados según el análisis de sentimiento y datos fundamentales.
- **Simulaciones de Monte Carlo**: Ejecuta simulaciones de Monte Carlo para evaluar el rendimiento del portafolio.

### Uso
1. Inicializa el optimizador con una lista de criptomonedas y una inversión inicial.
```python
crypto_ids = ['bitcoin', 'ethereum', 'cardano']
investment = 10000
optimizer = CryptoPortfolioOptimizer(crypto_ids, investment)
```
2. Obtén datos históricos de precios.
```python
optimizer.fetch_historical_data(days=365, retry_delay=60)
```
3. Realiza análisis de sentimiento y fundamental.
```python
optimizer.integrate_additional_analysis()
```
4. Optimiza el portafolio usando distintos métodos.
```python
result = optimizer.optimize_portfolio(min_weight=0.05, max_weight=0.4, method='max_return')
```
5. Ejecuta simulaciones de Monte Carlo para evaluar el rendimiento.
```python
monte_carlo_results = optimizer.run_monte_carlo_simulation(num_simulations=1000, days=252)
```

## Instalación
Para configurar el proyecto, instala las dependencias requeridas ejecutando:

```bash
pip install -r requirements.txt
```

### requirements.txt
```plaintext
numpy
pandas
pulp
requests
plotly
matplotlib
praw
nltk
tabulate
```
