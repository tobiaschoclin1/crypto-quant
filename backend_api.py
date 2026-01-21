from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# --- NUEVA LÓGICA DE DATOS ---
def obtener_datos(limit=1000):
    # Lista de espejos para evitar bloqueos
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines"
    ]
    
    params = {"symbol": "BTCUSDT", "interval": "15m", "limit": limit}
    
    for url in urls:
        try:
            response = requests.get(url, params=params, timeout=2)
            if response.status_code == 200:
                data = response.json()
                # Extraemos Precio (col 4) y Volumen (col 5)
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'etc', 'etc', 'etc', 'etc', 'etc', 'etc'])
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                return df
        except:
            continue
            
    return pd.DataFrame() # Retorno vacío si falla

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
    except:
        return ""

def calcular_todo():
    df = obtener_datos()
    if df.empty: return {"error": "Sin datos", "precio": 0, "score": 0, "decision": "Error", "detalles": [], "grafico_img": ""}

    # Variables principales
    precios = df['close'].values
    volumenes = df['volume'].values
    precio_actual = precios[-1]
    
    # 1. Indicadores Técnicos
    sma_100 = df["close"].rolling(window=100).mean().values
    sma_vol_20 = df["volume"].rolling(window=20).mean().values # Volumen promedio reciente
    rsi = calcular_rsi(df["close"]).values
    
    # Derivadas (Velocidad y Aceleración)
    suave = gaussian_filter1d(precios, sigma=1)
    vel = np.gradient(suave)
    
    # --- SISTEMA DE PUNTUACIÓN 2.0 ---
    score = 0
    razones = []
    
    # A. Tendencia (Peso: 3)
    sma_val = sma_100[-1] if not np.isnan(sma_100[-1]) else precio_actual
    if precio_actual > sma_val:
        score += 3
        razones.append("Tendencia Alcista ✅")
    else:
        razones.append("Tendencia Bajista ❌")
    
    # B. RSI - El Termómetro (Peso: Variado)
    rsi_val = rsi[-1] if not np.isnan(rsi[-1]) else 50
    if 30 <= rsi_val <= 70:
        score += 2
        razones.append(f"RSI Saludable ({int(rsi_val)}) ✅")
    elif rsi_val > 70:
        score -= 2 # PENALIZACIÓN
        razones.append(f"RSI Sobrecompra ({int(rsi_val)}) ⚠️")
    elif rsi_val < 30:
        score += 3 # Oportunidad de rebote
        razones.append(f"RSI Sobreventa ({int(rsi_val)}) 🚀")

    # C. Validación por Volumen (Peso: 2)
    vol_actual = volumenes[-1]
    vol_promedio = sma_vol_20[-1] if not np.isnan(sma_vol_20[-1]) else vol_actual
    
    if vol_actual > vol_promedio:
        score += 2
        razones.append("Volumen Alto (Confirmación) ✅")
    else:
        razones.append("Volumen Bajo (Débil) ⚠️")

    # D. Velocidad (Momentum)
    if vel[-1] > 0:
        score += 1
        razones.append("Impulso Positivo ✅")
    else:
        score -= 1
        razones.append("Impulso Negativo ❌")
        
    # --- DECISIÓN FINAL ---
    # Normalizamos el score para que esté entre 0 y 10 aprox
    score = max(0, min(10, score))
    
    if score >= 8: decision = "COMPRA FUERTE 🚀"
    elif score >= 6: decision = "COMPRA MODERADA 🟢"
    elif score >= 4: decision = "OBSERVAR 👀"
    elif score >= 2: decision = "VENTA 🔴"
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)