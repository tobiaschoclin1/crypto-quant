import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

def obtener_precios_binance(symbol="BTCUSDT", limit=10):
    """
    Obtiene los últimos 'limit' precios de cierre de Binance.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": limit}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Binance devuelve: [time, open, high, low, close, volume, ...]
    # Nos interesa el cierre (índice 4)
    tiempos = np.arange(limit) # 0, 1, 2...
    precios = np.array([float(x[4]) for x in data])
    
    return tiempos, precios

# 1. Obtener datos reales
print("Consultando a Binance...")
x_puntos, y_puntos = obtener_precios_binance()

# 2. Aplicar Lagrange (Matemática Simbólica a Numérica)
polinomio = lagrange(x_puntos, y_puntos)

print(f"\nÚltimos 10 precios de BTC: {y_puntos}")
print("\nPolinomio generado (simplificado):")
print(np.poly1d(polinomio))

# 3. Graficar
x_suave = np.linspace(0, 9, 100)
y_suave = polinomio(x_suave)  # <--- Aquí estaba el error, ahora dice 'polinomio'

plt.figure(figsize=(10, 6))
plt.scatter(x_puntos, y_puntos, color='red', label='Datos Binance (Discretos)')
plt.plot(x_suave, y_suave, label='Interpolación Lagrange (Continuo)')
plt.title("Convirtiendo Precio de Bitcoin en Función Matemática")
plt.xlabel("Minutos (0 = hace 10 min, 9 = ahora)")
plt.ylabel("Precio USDT")
plt.legend()
plt.grid(True)
plt.show()