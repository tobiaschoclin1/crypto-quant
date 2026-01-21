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

# --- TUS CREDENCIALES TELEGRAM ---
TELEGRAM_TOKEN = "8352173352:AAF1EuGRmTdbyDD_edQodfp3UPPeTWqqgwA" 
TELEGRAM_CHAT_ID = "793016927"
# ---------------------------------

# --- BILLETERA DE PAPER TRADING (SIMULADOR) ---
# OJO: En Render gratuito, esta memoria se reinicia si el servidor se apaga.
# Para pruebas de 24-48hs funciona bien con UptimeRobot.
portfolio = {
    "usdt": 1000.0,      # Capital inicial ficticio
    "btc": 0.0,          # Cantidad de BTC
    "in_market": False,  # ¿Estamos comprados?
    "entry_price": 0.0,  # Precio al que compramos
    "last_result": 0.0,   # Última ganancia/pérdida
    "trades_count": 0    # Cantidad de operaciones
}

ultima_alerta = ""

def enviar_telegram(mensaje):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=5)
    except: pass

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: No se encuentra index.html</h1>"

def obtener_datos(limit=1000):
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines"
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

def ejecutar_simulacion(precio_actual, score, razon_principal):
    global portfolio
    
    # REGLA DE COMPRA: Score >= 8 y tenemos USDT
    if not portfolio["in_market"] and score >= 8:
        portfolio["btc"] = portfolio["usdt"] / precio_actual
        portfolio["entry_price"] = precio_actual
        portfolio["usdt"] = 0
        portfolio["in_market"] = True
        enviar_telegram(f"🔵 *SIMULACIÓN: COMPRA*\nEntramos en ${precio_actual:,.2f}\nCapital: $1,000 USDT")
        
    # REGLA DE VENTA: Score <= 4 y tenemos BTC
    elif portfolio["in_market"] and score <= 4:
        # Vendemos todo
        valor_venta = portfolio["btc"] * precio_actual
        profit_usdt = valor_venta - 1000 # Asumiendo base 1000 para cálculo rápido de este trade (simplificado)
        # Ajuste real del portfolio
        # Recuperamos el valor base anterior:
        costo_original = portfolio["btc"] * portfolio["entry_price"]
        ganancia_real = valor_venta - costo_original
        
        portfolio["usdt"] = valor_venta
        portfolio["btc"] = 0
        portfolio["in_market"] = False
        portfolio["last_result"] = ganancia_real
        portfolio["trades_count"] += 1
        
        icono = "✅" if ganancia_real > 0 else "❌"
        enviar_telegram(f"🟠 *SIMULACIÓN: VENTA*\nSalimos en ${precio_actual:,.2f}\nResultado: {icono} ${ganancia_real:,.2f}")

def calcular_todo():
    global ultima_alerta
    df = obtener_datos()
    if df.empty: return {"error": "Sin datos", "precio": 0, "score": 0, "decision": "Error"}

    precios = df['c'].values
    volumenes = df['v'].values
    precio_actual = precios[-1]
    
    # Indicadores
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
    vol_act = volumenes[-1]
    vol_avg = sma_vol[-1] if not np.isnan(sma_vol[-1]) else vol_act
    if vol_act > vol_avg: score += 2; razones.append("Volumen Alto ✅")
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

    # --- EJECUTAR SIMULADOR ---
    ejecutar_simulacion(precio_actual, score, razones[0])

    # --- ALERTAS CLÁSICAS ---
    clave_alerta = f"{decision}-{int(precio_actual/100)}"
    if clave_alerta != ultima_alerta:
        if score >= 8:
            enviar_telegram(f"🚀 *ALERTA: OPORTUNIDAD*\nPrecio: ${precio_actual:,.2f}\nScore: {score}/10")
            ultima_alerta = clave_alerta
        elif score <= 2:
            enviar_telegram(f"🩸 *ALERTA: PELIGRO*\nPrecio: ${precio_actual:,.2f}\nScore: {score}/10")
            ultima_alerta = clave_alerta

    # Calcular valor total del portfolio para enviarlo al frontend
    total_balance = portfolio["usdt"]
    if portfolio["in_market"]:
        total_balance = portfolio["btc"] * precio_actual

    return {
        "precio": precio_actual,
        "score": score,
        "decision": decision,
        "detalles": razones,
        "grafico_img": generar_grafico_base64(precios),
        # Datos del Simulador para el Frontend
        "simulacion": {
            "balance": total_balance,
            "invertido": portfolio["in_market"],
            "precio_entrada": portfolio["entry_price"],
            "trades": portfolio["trades_count"]
        }
    }

@app.get("/analisis")
def get_analisis():
    return calcular_todo()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)