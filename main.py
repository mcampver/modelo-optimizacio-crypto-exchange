import os
from dotenv import load_dotenv

# Our modules
from src.modules.simplex import CryptoPortfolioOptimizer
from src.display.display_crypto_info import display_crypto_info
from src.display.display_optimization_results import display_optimization_results

def main():
    # Cargar las llaves de los servicios
    load_dotenv("src/config/keys.env")
    
    crypto_ids = ['bitcoin', 'ethereum', 'ripple', 'tether', 'solana', 'binancecoin', 'usd-coin', 'dogecoin', 'cardano', 'staked-ether']
    investment = 10000
    
    optimizer = CryptoPortfolioOptimizer(crypto_ids, investment)
    
    try:
        optimizer.fetch_historical_data(days=365, retry_delay=60)

        # Optimización con integración de análisis adicional (sentimiento y fundamental)
        # result_adjusted = optimizer.optimize_portfolio(min_weight=0.05, max_weight=0.4, method='max_return')
        # print("Optimización con análisis adicional:")
        # print(f"Pesos del Portafolio: {result_adjusted['weights']}")
        # print(f"Rentabilidad Esperada (ajustada): {result_adjusted['expected_return']:.2f}")
        # print(f"Riesgo del Portafolio (Volatilidad): {result_adjusted['portfolio_risk'] if result_adjusted['portfolio_risk'] is not None else 'No disponible'}")
        # print("Puntajes de Sentimiento:", result_adjusted['sentiment_scores'])
        # print("Datos Fundamentales:", result_adjusted['fundamental_data'])
        # print()
        methods = ['max_return', 'min_variance', 'max_sharpe']
        optimization_results = []
        
        for method in methods:
            print(f"\nEjecutando optimización usando el método: {method}")
            result = optimizer.optimize_portfolio(min_weight=0.05, max_weight=0.4, method=method)
            optimization_results.append(result)

        display_crypto_info(optimizer)
        
        # Mostrar los resultados de las optimizaciones en una tabla
        display_optimization_results(optimization_results, methods)
        # optimizer.fetch_historical_data(days=365, retry_delay=60)
        #  # Optimización de Maximización de Retorno (Método por defecto)
        # result_max_return = optimizer.optimize_portfolio(min_weight=0.05, max_weight=0.4, method='max_return')
        # print("Optimización de Maximización de Retorno:")
        # print(f"Pesos del Portafolio: {result_max_return['weights']}")
        # print(f"Rentabilidad Esperada: {result_max_return['expected_return']:.2f}")
        # print(f"Riesgo del Portafolio (Volatilidad): {result_max_return['portfolio_risk'] if result_max_return['portfolio_risk'] is not None else 'No disponible'}")
        # print()

        # # Optimización de Mínima Varianza
        # result_min_variance = optimizer.optimize_portfolio(min_weight=0.05, max_weight=0.4, method='min_variance')
        # print("Optimización de Mínima Varianza:")
        # print(f"Pesos del Portafolio: {result_min_variance['weights']}")
        # print(f"Rentabilidad Esperada: {result_min_variance['expected_return']}")
        # print(f"Riesgo del Portafolio (Volatilidad): {result_min_variance['portfolio_risk'] if result_min_variance['portfolio_risk'] is not None else 'No disponible'}")
        # print()

        # # Optimización de Maximización del Sharpe Ratio
        # result_max_sharpe = optimizer.optimize_portfolio(min_weight=0.05, max_weight=0.4, method='max_sharpe')
        # print("Optimización de Maximización del Sharpe Ratio:")
        # print(f"Pesos del Portafolio: {result_max_sharpe['weights']}")
        # print(f"Rentabilidad Esperada: {result_max_sharpe['expected_return']:.2f}")
        # print(f"Riesgo del Portafolio (Volatilidad): {result_max_sharpe['portfolio_risk'] if result_max_sharpe['portfolio_risk'] is not None else 'No disponible'}")
        # print()

        # invested_values = optimizer.get_invested_values()
        # print("Valores invertidos en cada criptomoneda:")
        # for crypto, value in invested_values.items():
        #     print(f"{crypto}: ${value:.2f}")
        
        # Ejecutar Monte Carlo
        # monte_carlo_results = optimizer.run_monte_carlo_simulation(num_simulations=1000, days=252)
        
        # print(monte_carlo_results)
        
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()
    