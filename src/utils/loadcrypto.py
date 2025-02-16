import requests
import json
import os

# Función para obtener las principales criptomonedas de la API
def fetch_top_cryptos(api_url="https://api.coingecko.com/api/v3/coins/markets", limit=10, currency="usd"):
    # Obtener las 10 criptomonedas principales por volumen de mercado
    params = {
     'vs_currency': currency,
     'order': 'price_change_percentage_24h',  # Ordenar por volatilidad (cambio porcentual en 24 horas)
     'per_page': limit,
     'page': 1
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        cryptos = response.json()
        
        # Extraer solo los ID de las criptomonedas
        crypto_ids = [crypto['id'] for crypto in cryptos]
        return crypto_ids
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener datos de la API: {e}")
        return []

# Función para guardar los datos de criptomonedas en un archivo
def save_cryptos_to_file(filename="src/data/cryptos.json", limit=10):
    crypto_ids = fetch_top_cryptos(limit=limit)
    
    if not crypto_ids:
        print("No se pudieron obtener criptomonedas.")
        return
    
    # Verificar si el archivo ya existe
    if os.path.exists(filename):
        respuesta = input(f"El archivo {filename} ya existe. ¿Deseas sobrescribirlo? (s/n): ")
        if respuesta.lower() not in ['s', 'si']:
            print("Operación cancelada.")
            return
    else:
        # Asegurarse de que la carpeta exista
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Guardar los IDs de las criptomonedas en un archivo JSON
    with open(filename, 'w') as f:
        json.dump(crypto_ids, f, indent=4)
    print(f"Criptomonedas guardadas en {filename}")

# Función para cargar criptomonedas desde el archivo
def load_cryptos_from_file(filename="src/data/cryptos.json"):
    if not os.path.exists(filename):
        print(f"Archivo {filename} no encontrado. Realizando la descarga de criptomonedas...")
        save_cryptos_to_file(filename=filename)
    
    with open(filename, 'r') as f:
        crypto_ids = json.load(f)
    return crypto_ids

# Función principal
def main():
    print("Directorio de trabajo actual:", os.getcwd())
    save_cryptos_to_file()
    crypto_ids = load_cryptos_from_file()
    print(f"Criptomonedas cargadas: {crypto_ids}")

if __name__ == "__main__":
    main()
