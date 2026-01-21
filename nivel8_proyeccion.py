import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. MATEMÁTICA (Modelo Logístico) ---
# Esta es la misma fórmula que usaste en Simulador 2 para poblaciones
def modelo_logistico(t, L, k, t0):
    """
    L: Capacidad de carga (Precio Máximo Teórico de este impulso)
    k: Tasa de crecimiento (Virulencia de la compra)
    t0: Punto de inflexión (Momento de máxima aceleración)
    """
    return L / (1 + np.exp(-k * (t - t0)))

# --- 2. DATOS ---
print("Obteniendo datos macro...")
# Usamos velas de 1 Hora (1h) para ver la "Big Picture" de la tendencia actual
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 500}
data = requests.get(url, params=params).json()

precios = np.array([float(x[4]) for x in data])
# Creamos un eje de tiempo simple (0, 1, 2, ... 500)
x_tiempo = np.arange(len(precios))

# --- 3. INGENIERÍA INVERSA (Curve Fitting) ---
# Intentamos forzar a la matemática a encontrar L, k y t0 que mejor se ajusten al precio real
# Damos unos "valores iniciales" (p0) para ayudar al algoritmo a no perderse
precio_actual = precios[-1]
p0 = [precio_actual * 1.1, 0.05, len(precios)/2] 

try:
    # curve_fit devuelve los parámetros óptimos (popt) y la covarianza (pcov)
    popt, pcov = curve_fit(modelo_logistico, x_tiempo, precios, p0=p0, maxfev=10000)
    
    L_optimo, k_optimo, t0_optimo = popt
    
    # --- 4. PROYECCIÓN FUTURA ---
    # Proyectamos 50 horas al futuro
    x_futuro = np.arange(len(precios) + 50)
    y_proyectada = modelo_logistico(x_futuro, *popt)
    
    print("\n" + "="*30)
    print("RESULTADOS DEL MODELO LOGÍSTICO")
    print(f"Precio Actual:      ${precio_actual:.2f}")
    print(f"Techo Proyectado (L): ${L_optimo:.2f}")
    print(f"Velocidad (k):      {k_optimo:.4f}")
    print("="*30)
    
    # --- 5. GRAFICAR ---
    plt.figure(figsize=(12, 6))
    
    # Datos reales
    plt.plot(x_tiempo, precios, label="Precio Real (BTC)", color="black", alpha=0.5)
    
    # Curva ajustada (S-Curve)
    plt.plot(x_futuro, y_proyectada, label=f"Modelo Logístico (Target: ${L_optimo:.0f})", color="red", linewidth=2, linestyle="--")
    
    # Línea del techo teórico
    plt.axhline(L_optimo, color="green", linestyle=":", label="Capacidad de Carga (Resistencia)")
    
    plt.title("Proyección de Saturación de Tendencia (BTC)")
    plt.xlabel("Horas")
    plt.ylabel("Precio USDT")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

except Exception as e:
    print(f"\nError: No se pudo ajustar una curva S a estos datos.")
    print(f"Razón técnica: El precio quizás está cayendo o es demasiado lineal.")
    print(f"Detalle: {e}")