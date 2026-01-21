from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg') # Vital para servidores sin pantalla
import matplotlib.pyplot as plt
import io
import base64
import os

app = FastAPI()

# Configuración de CORS (Para seguridad estándar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NUEVO: SERVIR EL HTML ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    # Leemos el archivo index.html y lo enviamos al navegador
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# --- FUNCIONES MATEMÁTICAS (Tu motor) ---
def obtener_datos(limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "15m", "limit": limit}
    try:
        data = requests.get(url, params=params).json()
        precios = np.array([float(x[4]) for x in data])
        return precios
    except:
        return np.array([])

def generar_grafico_base64(precios):
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
        data_plot = precios[-50:]
        x = range(len(data_plot))
        ax.plot(x, data_plot, color='#22d3ee', linewidth=2.5)
        ax.fill_between(x, data_plot, data_plot.min(), color='#22d3ee', alpha=0.1)
        ax.axis('off')
        ax.grid(False)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        raw_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{raw_b64}"
    except Exception as e:
        return ""

def calcular_todo():
    precios = obtener_datos()
    if len(precios) == 0: return {"error": "Sin datos", "precio": 0, "score": 0, "decision": "Error", "detalles": [], "grafico_img": ""}

    precio_actual = precios[-1]
    df = pd.DataFrame({"close": precios})
    sma_100 = df["close"].rolling(window=100).mean().values
    suave = gaussian_filter1d(precios, sigma=1)
    vel = np.gradient(suave)
    acel = np.gradient(vel)
    
    score = 0
    razones = []
    
    # Lógica Quant Simplificada
    sma_val = sma_100[-1] if not np.isnan(sma_100[-1]) else precio_actual
    
    if precio_actual > sma_val: score += 3; razones.append("Tendencia Alcista ✅")
    else: razones.append("Tendencia Bajista ❌")
    
    if vel[-1] > 0: score += 3; razones.append("Velocidad Positiva ✅")
    else: razones.append("Velocidad Negativa ❌")
        
    if acel[-1] > 0: score += 2; razones.append("Ganando Fuerza ✅")
    else: razones.append("Perdiendo Fuerza ⚠️")
    
    # Ajuste final
    if score >= 5: score += 2; razones.append("Estructura Fuerte ✅")
    else: razones.append("Estructura Débil ❌")

    if score >= 8: decision = "COMPRA FUERTE 🚀"
    elif score >= 5: decision = "OBSERVAR 👀"
    elif score >= 3: decision = "ESPERAR ✋"
    else: decision = "VENTA FUERTE 🩸"
    
    return {
        "precio": precio_actual,
        "score": score,
        "decision": decision,
        "detalles": razones,
        "grafico_img": generar_grafico_base64(precios)
    }

@app.get("/analisis")
def get_analisis():
    return calcular_todo()

# Configuración para que Render sepa cómo correrlo
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)