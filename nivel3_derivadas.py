import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- 1. OBTENER DATOS (SENSOR) ---
def obtener_datos(symbol="BTCUSDT", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    
    # Solo nos interesa el precio de cierre (Close)
    precios = np.array([float(x[4]) for x in data])
    return precios

print("Leyendo sensores de Binance...")
precios_raw = obtener_datos()

# --- 2. PROCESAMIENTO DE SEÑAL (FILTRO + CÁLCULO) ---
# Aplicamos un filtro Gaussiano (Sigma=2) para eliminar el ruido de alta frecuencia
# Esto es vital: sin esto, la derivada se vuelve loca con cada pequeño tick.
precios_suaves = gaussian_filter1d(precios_raw, sigma=2)

# Primera Derivada (Velocidad): ¿Qué tan rápido cambia el precio?
# Unidades: USDT por minuto
velocidad = np.gradient(precios_suaves)

# Segunda Derivada (Aceleración): ¿La velocidad está aumentando o disminuyendo?
# Unidades: USDT por minuto^2
aceleracion = np.gradient(velocidad)

# --- 3. VISUALIZACIÓN DE INGENIERÍA ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Panel 1: Posición (Precio)
ax1.plot(precios_raw, color='gray', alpha=0.3, label='Precio Crudo (Ruido)')
ax1.plot(precios_suaves, color='blue', linewidth=2, label='Tendencia (Filtrada)')
ax1.set_ylabel('Precio (USDT)')
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_title(f'Análisis Dinámico de BTCUSDT (Últimos {len(precios_raw)} min)')

# Panel 2: Velocidad (Momentum)
ax2.plot(velocidad, color='orange', label='Velocidad (dP/dt)')
ax2.axhline(0, color='black', linestyle='--', alpha=0.5) # Línea cero
ax2.fill_between(range(len(velocidad)), velocidad, 0, where=(velocidad > 0), color='green', alpha=0.1)
ax2.fill_between(range(len(velocidad)), velocidad, 0, where=(velocidad < 0), color='red', alpha=0.1)
ax2.set_ylabel('Velocidad ($/min)')
ax2.legend(loc='upper left')
ax2.grid(True)

# Panel 3: Aceleración (Fuerza)
ax3.plot(aceleracion, color='purple', label='Aceleración (d²P/dt²)')
ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
ax3.set_ylabel('Aceleración')
ax3.set_xlabel('Tiempo (minutos)')
ax3.legend(loc='upper left')
ax3.grid(True)

plt.tight_layout()
plt.show()