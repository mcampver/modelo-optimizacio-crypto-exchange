import pandas as pd
from tabulate import tabulate

# Our Modules
from src.modules.simplex import CryptoPortfolioOptimizer

def display_crypto_info(optimizer: CryptoPortfolioOptimizer):
      data = []
      for crypto in optimizer.available_cryptos:
        sentiment = optimizer.sentiment_scores.get(crypto, "No disponible")
        fundamentals = optimizer.fundamental_data.get(crypto, {})
        market_cap = fundamentals.get("market_cap", "No disponible")
        total_volume = fundamentals.get("total_volume", "No disponible")
        circulating_supply = fundamentals.get("circulating_supply", "No disponible")
        data.append({
            "Criptomoneda": crypto,
            "Sentimiento": round(sentiment, 5) if isinstance(sentiment, (int, float)) else sentiment,
            "Market Cap": market_cap,
            "Volumen Total": total_volume,
            "Suministro Circulante": circulating_supply
        })
      df = pd.DataFrame(data)
      print("\nInformación de Criptomonedas:")
      print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
