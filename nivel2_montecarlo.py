import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. CONFIGURACIÓN ---
SYMBOL = "BTCUSDT"
INTERVALO = "1m"    # Velas de 1 minuto
LIMIT = 100         # Usamos 100 minutos de historia para calcular la "personalidad" del precio
SIMULACIONES = 50   # Cuántos futuros alternativos vamos a imaginar
PASOS_FUTUROS = 20  # Cuántos minutos al futuro queremos ver

def obtener_datos(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    
    # Creamos un DataFrame de Pandas (como un Excel con esteroides)
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "vol", "x", "y", "z", "w", "a", "b"])
    df["close"] = df["close"].astype(float)
    return df["close"]

# --- 2. EL MOTOR MATEMÁTICO ---
print(f"Descargando datos de {SYMBOL}...")
precios_hist = obtener_datos(SYMBOL, INTERVALO, LIMIT)

# Calculamos los "Retornos Logarítmicos" 
# (Es la forma matemática de medir cuánto cambió el precio % paso a paso)
log_returns = np.log(1 + precios_hist.pct_change())

# Estadísticas clave (La "personalidad" de Bitcoin hoy)
u = log_returns.mean() # Media (Drift/Tendencia)
var = log_returns.var() # Varianza (Riesgo)
drift = u - (0.5 * var) # Ajuste técnico para movimiento Browniano
stdev = log_returns.std() # Desviación estándar (Volatilidad pura)

# --- 3. SIMULACIÓN MONTE CARLO ---
# Generamos números aleatorios con distribución normal (Campana de Gauss)
# Z es una matriz de [pasos_futuros x simulaciones]
Z = norm.ppf(np.random.rand(PASOS_FUTUROS, SIMULACIONES))

# Fórmula del Movimiento Browniano Geométrico (GBM)
# Precio_futuro = Precio_actual * e^(drift + vol * aleatoriedad)
retornos_diarios = np.exp(drift + stdev * Z)

# Proyectamos precios
ultimo_precio = precios_hist.iloc[-1]
lista_precios = np.zeros_like(retornos_diarios)
lista_precios[0] = ultimo_precio

for t in range(1, PASOS_FUTUROS):
    lista_precios[t] = lista_precios[t - 1] * retornos_diarios[t]

# --- 4. VISUALIZACIÓN ---
plt.figure(figsize=(10, 6))

# Graficamos la historia (Línea negra)
plt.plot(range(LIMIT), precios_hist, color='black', label='Historia Real', linewidth=2)

# Graficamos las simulaciones (Líneas de colores)
rango_futuro = range(LIMIT, LIMIT + PASOS_FUTUROS)
plt.plot(rango_futuro, lista_precios, alpha=0.3) 

plt.title(f"Monte Carlo: {SIMULACIONES} futuros posibles para {SYMBOL}")
plt.xlabel("Minutos")
plt.ylabel("Precio (USDT)")
plt.axvline(x=LIMIT, color='red', linestyle='--', label='Ahora')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Precio actual: {ultimo_precio:.2f}")
print(f"Volatilidad detectada: {stdev:.5f}")