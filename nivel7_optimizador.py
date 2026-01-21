import requests
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# --- 1. MOTOR DE SIMULACIÓN (Parametrizado) ---
def correr_simulacion(precios, sma_window, sigma):
    # Calculamos indicadores con los parámetros recibidos
    df_temp = pd.DataFrame({"close": precios})
    sma = df_temp["close"].rolling(window=sma_window).mean().values
    
    # Suavizado de la curva
    suave = gaussian_filter1d(precios, sigma=sigma)
    velocidad = np.gradient(suave)
    aceleracion = np.gradient(velocidad)
    
    # Billetera Temporal (Reinicia en cada simulación)
    usdt = 1000
    btc = 0
    en_posicion = False
    precio_entrada = 0
    trades = 0
    
    # Recorremos la historia
    # Empezamos después del SMA más grande posible para no tener errores
    start_index = 100 
    
    for t in range(start_index, len(precios)):
        precio = precios[t]
        sma_val = sma[t]
        
        # Física actual
        vel_actual = velocidad[t]
        vel_anterior = velocidad[t-1]
        acel_actual = aceleracion[t]
        
        # --- ESTRATEGIA PURA ---
        # 1. COMPRA
        if not en_posicion:
            if (vel_anterior < 0 and vel_actual > 0) and (precio > sma_val):
                btc = usdt / precio
                usdt = 0
                en_posicion = True
                precio_entrada = precio
                trades += 1
                
        # 2. VENTA
        elif en_posicion:
            motivo = None
            # Stop Dinámico (SMA)
            if precio < sma_val: motivo = "SMA"
            # Divergencia
            elif (vel_actual < vel_anterior) and (acel_actual < -0.5): motivo = "Div"
            
            if motivo:
                usdt = btc * precio
                btc = 0
                en_posicion = False
    
    # Calcular saldo final
    saldo_final = (btc * precios[-1]) if en_posicion else usdt
    rendimiento = ((saldo_final - 1000) / 1000) * 100
    return rendimiento, trades

# --- 2. OBTENCIÓN DE DATOS ---
print("--- INICIANDO OPTIMIZADOR DE PARÁMETROS ---")
print("Descargando 1000 velas de 15m (aprox. 10 días)...")
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 1000}
data = requests.get(url, params=params).json()
precios_raw = np.array([float(x[4]) for x in data])

# --- 3. BUCLE DE OPTIMIZACIÓN (GRID SEARCH) ---
mejores_params = {"rend": -999, "sma": 0, "sigma": 0, "trades": 0}
resultados = []

print(f"Probando combinaciones...")
print(f"{'SMA':<5} | {'SIGMA':<5} | {'RENDIMIENTO':<12} | {'TRADES':<5}")
print("-" * 40)

# Probamos SMA de 20 a 100 (saltando de 10 en 10)
for sma_test in range(20, 110, 10):
    # Probamos Sigma de 1 a 4
    for sigma_test in range(1, 5):
        
        rend, num_trades = correr_simulacion(precios_raw, sma_test, sigma_test)
        
        # Filtro de calidad: Ignoramos si hizo menos de 3 operaciones (poco confiable)
        if num_trades > 3:
            print(f"{sma_test:<5} | {sigma_test:<5} | {rend:>10.2f}% | {num_trades:<5}")
            
            if rend > mejores_params["rend"]:
                mejores_params = {
                    "rend": rend,
                    "sma": sma_test,
                    "sigma": sigma_test,
                    "trades": num_trades
                }

print("-" * 40)
print("\n🏆 CONFIGURACIÓN GANADORA 🏆")
print(f"Media Móvil (SMA): {mejores_params['sma']}")
print(f"Suavizado (Sigma): {mejores_params['sigma']}")
print(f"Rendimiento Total: {mejores_params['rend']:.2f}%")
print(f"Operaciones:       {mejores_params['trades']}")