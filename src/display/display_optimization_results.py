from typing import List, Dict
import pandas as pd
from tabulate import tabulate

def display_optimization_results(results: List[Dict], methods: List[str]):
     table_data = []
     for method, result in zip(methods, results):
        pesos = ", ".join([f"{k}: {v:.2f}" for k, v in result["weights"].items()])
        table_data.append({
            "Método": method,
            "Pesos del Portafolio": pesos,
            "Rentabilidad Esperada": f"{result['expected_return']:.2f}",
            "Riesgo (Volatilidad)": f"{result['portfolio_risk']:.2f}" if result['portfolio_risk'] is not None else "No disponible"
        })
     df = pd.DataFrame(table_data)
     print("\nResultados de Optimización:")
     print(tabulate(df, headers='keys', tablefmt='psql', showindex=False)) 
