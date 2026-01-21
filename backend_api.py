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
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CONFIGURACIÓN TELEGRAM ---
TELEGRAM_TOKEN = "8352173352:AAF1EuGRmTdbyDD_edQodfp3UPPeTWqqgwA" 
TELEGRAM_CHAT_ID = "793016927"   # Ej: 12345678 (Pídeselo a @userinfobot)
# ------------------------------

# Variable global para no spamear alertas repetidas
ultima_alerta = ""

def enviar_telegram(mensaje):
    if "PEGA_AQUI" in TELEGRAM_CHAT_ID: return # No configurado
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=5)
    except:
        pass

@app.get("/", response_class=HTMLResponse)
def read_root():
    # Intenta leer index.html, si falla crea uno básico de emergencia
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: No se encuentra index.html</h1>"

def obtener_datos(limit=1000):
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
            r = requests.get(url, params=params, timeout=2)
            if r.status_code == 200:
                data = r.json()
                df = pd.DataFrame(data, columns=['t', 'o', 'h', 'l', 'c', 'v', 'x', 'x', 'x', 'x', 'x', 'x'])
                df['c'] = df['c'].astype(float)
                df['v'] = df['v'].astype(float)
                return df
        except: continue
    return pd.DataFrame()

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
        data = precios[-50:]
        x = range(len(data))
        ax.plot(x, data, color='#22d3ee', linewidth=2.5)
        ax.fill_between(x, data, data.min(), color='#22d3ee', alpha=0.1)
        ax.axis('off'); ax.grid(False)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except: return ""

def calcular_todo():
    global ultima_alerta
    df = obtener_datos()
    if df.empty: return {"error": "Sin datos", "precio": 0, "score": 0, "decision": "Error", "detalles": [], "grafico_img": ""}

    precios = df['c'].values
    volumenes = df['v'].values
    precio_actual = precios[-1]
    
    # Cálculos
    sma_100 = df["c"].rolling(window=100).mean().values
    sma_vol = df["v"].rolling(window=20).mean().values
    rsi = calcular_rsi(df["c"]).values
    suave = gaussian_filter1d(precios, sigma=1)
    vel = np.gradient(suave)

    score = 0
    razones = []

    # 1. Tendencia
    sma_val = sma_100[-1] if not np.isnan(sma_100[-1]) else precio_actual
    if precio_actual > sma_val: score += 3; razones.append("Tendencia Alcista ✅")
    else: razones.append("Tendencia Bajista ❌")

    # 2. RSI
    rsi_val = rsi[-1] if not np.isnan(rsi[-1]) else 50
    if 30 <= rsi_val <= 70: score += 2; razones.append(f"RSI Neutro ({int(rsi_val)}) ✅")
    elif rsi_val > 70: score -= 2; razones.append(f"RSI Sobrecompra ({int(rsi_val)}) ⚠️")
    elif rsi_val < 30: score += 3; razones.append(f"RSI Sobreventa ({int(rsi_val)}) 🚀")

    # 3. Volumen
    vol_actual = volumenes[-1]
    sma_vol_val = sma_vol[-1] if not np.isnan(sma_vol[-1]) else vol_actual
    if vol_actual > sma_vol_val: score += 2; razones.append("Volumen Alto ✅")
    else: razones.append("Volumen Bajo ⚠️")

    # 4. Momentum
    if vel[-1] > 0: score += 1; razones.append("Impulso Positivo ✅")
    else: score -= 1; razones.append("Impulso Negativo ❌")

    score = max(0, min(10, score))

    if score >= 8: decision = "COMPRA FUERTE 🚀"
    elif score >= 6: decision = "COMPRA MODERADA 🟢"
    elif score >= 4: decision = "OBSERVAR 👀"
    elif score >= 2: decision = "VENTA 🔴"
    else: decision = "VENTA FUERTE 🩸"

    # --- LÓGICA DE ALERTAS TELEGRAM ---
    clave_alerta = f"{decision}-{int(precio_actual/100)}" 
    
    if clave_alerta != ultima_alerta:
        # Se activa si el score es extremo (>=8 o <=2)
        if score >= 8:
            enviar_telegram(f"🚀 *ALERTA QUANT: COMPRA*\nPrecio: ${precio_actual:,.2f}\nScore: {score}/10\nMotivo: {razones[1]}")
            ultima_alerta = clave_alerta
        elif score <= 2:
            enviar_telegram(f"🩸 *ALERTA QUANT: VENTA*\nPrecio: ${precio_actual:,.2f}\nScore: {score}/10\nMotivo: {razones[1]}")
            ultima_alerta = clave_alerta

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

@app.get("/test-telegram")
def test_telegram():
    enviar_telegram("✅ *Prueba de Sistema*\nEl bot Quant está conectado a tu celular.")
    return {"status": "Mensaje enviado (Revisa Telegram)"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)