import requests
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

# --- 1. MOTORES MATEMÁTICOS (Tus Módulos) ---

def obtener_datos(limit=1000):
    # Usamos 15m porque fue el ganador del Nivel 7
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "15m", "limit": limit}
    data = requests.get(url, params=params).json()
    precios = np.array([float(x[4]) for x in data])
    return precios

def analisis_tecnico(precios):
    # Parámetros Ganadores del Nivel 7: SMA=100, Sigma=1
    df = pd.DataFrame({"close": precios})
    sma_100 = df["close"].rolling(window=100).mean().values
    
    # Derivadas con Sigma=1
    suave = gaussian_filter1d(precios, sigma=1)
    vel = np.gradient(suave)
    acel = np.gradient(vel)
    
    return sma_100, vel, acel

def modelo_logistico(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def proyeccion_logistica(precios):
    # Usamos una ventana macro (últimos 500 periodos) para la curva S
    datos_macro = precios[-500:] 
    x = np.arange(len(datos_macro))
    p0 = [datos_macro[-1]*1.1, 0.01, len(datos_macro)/2]
    
    try:
        popt, _ = curve_fit(modelo_logistico, x, datos_macro, p0=p0, maxfev=10000)
        L_target = popt[0]
        return L_target
    except:
        return 0

# --- 2. EL CEREBRO DE DECISIÓN ---

print("\n--- INICIANDO SISTEMA DE INGENIERÍA QUANT ---")
print("Leyendo sensores de Binance...")

precios = obtener_datos()
precio_actual = precios[-1]

# Calculamos todo
sma, vel, acel = analisis_tecnico(precios)
target_logistico = proyeccion_logistica(precios)

sma_actual = sma[-1]
vel_actual = vel[-1]
acel_actual = acel[-1]

# --- 3. SISTEMA DE PUNTAJE (SCORING) ---
score = 0
razones = []

# Factor A: Tendencia (Peso: 3 puntos)
if precio_actual > sma_actual:
    score += 3
    razones.append(f"✅ TENDENCIA: Precio (${precio_actual:.0f}) sobre la media (${sma_actual:.0f})")
else:
    razones.append(f"❌ TENDENCIA: Precio debajo de la media (Mercado Bajista)")

# Factor B: Momentum (Peso: 3 puntos)
if vel_actual > 0:
    score += 3
    razones.append(f"✅ VELOCIDAD: Positiva (${vel_actual:.2f}/vela). El precio sube.")
else:
    razones.append(f"❌ VELOCIDAD: Negativa. El precio cae.")

# Factor C: Aceleración (Peso: 2 puntos)
if acel_actual > 0:
    score += 2
    razones.append(f"✅ FUERZA: Aceleración positiva. El movimiento gana fuerza.")
else:
    razones.append(f"⚠️ FUERZA: Aceleración negativa. El movimiento se cansa.")

# Factor D: Valoración Logística (Peso: 2 puntos)
# Si estamos más baratos que el Target Logístico, es bueno comprar
distancia_logistica = ((target_logistico - precio_actual) / precio_actual) * 100
if precio_actual < target_logistico:
    score += 2
    razones.append(f"✅ VALOR: Subvaluado un {distancia_logistica:.2f}% respecto al modelo (Target: ${target_logistico:.0f})")
else:
    razones.append(f"❌ VALOR: Sobrevaluado. Precio mayor al target logístico.")

# --- 4. REPORTE FINAL ---
print("\n" + "="*40)
print(f"REPORT QUANT: BTC/USDT (Intervalo 15m)")
print("="*40)
print(f"PRECIO ACTUAL: ${precio_actual:.2f}")
print("-" * 40)

for r in razones:
    print(r)

print("-" * 40)
print(f"PUNTAJE FINAL: {score}/10")
print("="*40)

if score >= 8:
    print("🚀 DECISIÓN: COMPRA FUERTE (Strong Buy)")
elif score >= 5:
    print("👀 DECISIÓN: OBSERVACIÓN / COMPRA DÉBIL")
elif score >= 3:
    print("✋ DECISIÓN: ESPERAR (Hold)")
else:
    print("🩸 DECISIÓN: VENTA FUERTE (Strong Sell)")
print("="*40)