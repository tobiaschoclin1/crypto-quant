import requests
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import time

# --- 1. CEREBRO DEL BOT (Lógica de Ingeniería) ---
def analizar_mercado(precios_raw):
    # Suavizamos la señal (Sigma=2) para no reaccionar al ruido
    precios = gaussian_filter1d(precios_raw, sigma=2)
    
    # Calculamos Velocidad y Aceleración
    velocidad = np.gradient(precios)
    aceleracion = np.gradient(velocidad)
    
    return precios, velocidad, aceleracion

# --- 2. SIMULACIÓN DE TIEMPO REAL ---
# En un bot real, esto sería un bucle infinito while True.
# Aquí simulamos minuto a minuto con los datos históricos para probar tu lógica.

print("--- INICIANDO SISTEMA DE VIGILANCIA QUANT ---")
print("Descargando datos de caja negra...")

# Bajamos 100 minutos de historia
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 100}
data = requests.get(url, params=params).json()
precios_cierre = np.array([float(x[4]) for x in data])

# Simulamos que el tiempo avanza minuto a minuto
# Empezamos desde el minuto 10 para tener algo de historia previa
for t in range(10, len(precios_cierre)):
    
    # Cortamos los datos hasta el momento "t" (simulando que es 'ahora')
    datos_hasta_ahora = precios_cierre[:t]
    
    # El cerebro analiza la situación actual
    p, v, a = analizar_mercado(datos_hasta_ahora)
    
    # Tomamos los valores actuales (el último del array)
    precio_actual = p[-1]
    vel_actual = v[-1]
    vel_anterior = v[-2]
    acel_actual = a[-1]
    
    # --- 3. REGLAS DE TRADING (TU LÓGICA) ---
    
    # REGLA A: Cruce por Cero (COMPRA)
    # Si la velocidad era negativa y ahora es positiva -> El precio empezó a subir
    if vel_anterior < 0 and vel_actual > 0:
        print(f"[MIN {t}] 🟢 SEÑAL DE COMPRA: El precio giró al alza. (Precio: {precio_actual:.2f})")
        
    # REGLA B: Divergencia Bajista (PELIGRO - Lo que viste en el min 50)
    # Si el precio sube... PERO la velocidad baja (aceleración negativa fuerte)
    elif (precio_actual > p[-2]) and (vel_actual < vel_anterior) and (acel_actual < -0.5):
        # El umbral -0.5 es para filtrar ruidos pequeños, queremos frenadas bruscas
        print(f"[MIN {t}] ⚠️ ALERTA DE DIVERGENCIA: Precio sube pero pierde fuerza. (Posible techo).")
        
    # REGLA C: Caída Libre (VENTA DE PÁNICO)
    # Si la velocidad es muy negativa
    elif vel_actual < -20:
        print(f"[MIN {t}] 🔴 CAÍDA FUERTE: El precio se desploma rápido.")

print("--- FIN DE LA SIMULACIÓN ---")