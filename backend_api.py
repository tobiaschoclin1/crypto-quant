from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
from datetime import datetime
import pytz 

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CREDENCIALES ---
TELEGRAM_TOKEN = "8352173352:AAF1EuGRmTdbyDD_edQodfp3UPPeTWqqgwA" 
TELEGRAM_CHAT_ID = "793016927"
GEMINI_API_KEY = "AIzaSyAp1WURjJ03HhdB8NzkO1Rhre5-FqRtFIA" 
# --------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]

# --- CONFIGURACIÓN DE TRADING ---
INITIAL_CAPITAL = 1000.0
BUY_AMOUNT = 200.0  # Compra de a $200 por vez

# Cartera (Ya no usamos "in_market", usamos saldos reales)
portfolios = {
    sym: {"usdt": INITIAL_CAPITAL, "coin": 0.0, "avg_price": 0.0, "trades": 0}
    for sym in SYMBOLS
}

last_signals = {sym: "NEUTRAL" for sym in SYMBOLS}
market_data_cache = {} 

def enviar_telegram(mensaje):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=3)
    except: pass

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: No se encuentra index.html</h1>"

@app.post("/chat")
async def chat_with_ai(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message", "")
        symbol = body.get("symbol", "BTCUSDT")
        api_key = GEMINI_API_KEY.strip()

        datos = market_data_cache.get(symbol, {})
        precio = datos.get("precio", 0)
        decision = datos.get("decision", "NEUTRAL")
        razones = datos.get("detalles", [])
        pf = datos.get("portfolio", {})

        contexto = f"""
        Eres un Asesor Financiero Crypto.
        Analiza {symbol}.
        
        DATOS TÉCNICOS:
        - Precio: ${precio:,.4f}
        - Señal Técnica: {decision}
        - Indicadores: {', '.join(razones)}
        
        MI CARTERA ACTUAL ({symbol}):
        - USDT Disponible: ${pf.get('usdt', 0):,.2f}
        - Crypto Tenencia: {pf.get('coin', 0):,.4f} {symbol.replace('USDT','')}
        - Precio Promedio Compra: ${pf.get('avg_price', 0):,.2f}
        
        Usuario: "{user_message}"
        Responde corto, útil y humano.
        """
        
        headers = {"Content-Type": "application/json"}
        payload = { "contents": [{ "parts": [{"text": contexto}] }] }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=5)
            if r.status_code == 200:
                return JSONResponse({"reply": r.json()['candidates'][0]['content']['parts'][0]['text']})
        except: pass
        
        return JSONResponse({"reply": "Estoy recalibrando mis sensores. Pregúntame en un momento."})

    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}"})

# --- LÓGICA DE MERCADO ---
def obtener_datos(symbol):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "15m", "limit": 200}
    try:
        r = requests.get(url, params=params, timeout=2)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=['t', 'o', 'h', 'l', 'c', 'v', 'x', 'x', 'x', 'x', 'x', 'x'])
            df['c'] = df['c'].astype(float)
            df['v'] = df['v'].astype(float)
            return df
    except: pass
    return pd.DataFrame()

def calcular_indicadores(df):
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['ema200'] = df['c'].ewm(span=200, adjust=False).mean()
    ema12 = df['c'].ewm(span=12, adjust=False).mean()
    ema26 = df['c'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df

def generar_grafico(df, symbol):
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        data = df['c'].tail(100)
        ema = df['ema200'].tail(100)
        x = range(len(data))
        ax.plot(x, data, color='#22d3ee', linewidth=2, label='Precio')
        ax.plot(x, ema, color='#fbbf24', linestyle='--', alpha=0.7, label='EMA 200')
        ax.fill_between(x, data, data.min(), color='#22d3ee', alpha=0.1)
        ax.axis('off'); ax.grid(False)
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except: return ""

def ejecutar_estrategia(symbol, df):
    global portfolios
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    precio = current['c']
    
    # ESTRATEGIA: Tendencia + Retroceso RSI + Cruce MACD
    tendencia_alcista = precio > current['ema200']
    cruce_macd_alcista = (prev['macd'] < prev['signal']) and (current['macd'] > current['signal'])
    rsi_bajo = current['rsi'] < 55 # Buscamos entrar en retrocesos
    
    # PUNTUACIÓN
    score = 5
    razones = []
    
    if tendencia_alcista: score += 2; razones.append("Tendencia Alcista ✅")
    else: score -= 2; razones.append("Tendencia Bajista ⚠️")
    
    if current['rsi'] > 70: score -= 2; razones.append("Sobrecompra ⚠️")
    elif current['rsi'] < 30: score += 2; razones.append("Sobreventa 🚀")
    
    if cruce_macd_alcista: score += 3; razones.append("Cruce MACD Alcista 🔥")
    
    score = max(0, min(10, score))
    
    # DECISIÓN BASE
    decision = "NEUTRAL"
    # Compramos si hay tendencia Y cruce O rsi muy bajo
    if tendencia_alcista and (cruce_macd_alcista or score >= 7): decision = "COMPRA"
    # Vendemos si perdemos tendencia o indicador de salida fuerte
    elif (not tendencia_alcista) or score <= 3: decision = "VENTA"
    
    # GESTIÓN DE CARTERA MIXTA
    pf = portfolios[symbol]
    
    # --- LÓGICA DE COMPRA (Acumulativa) ---
    # Compramos si la señal es COMPRA Y tenemos USDT suficiente
    if decision == "COMPRA" and pf["usdt"] >= BUY_AMOUNT:
        cantidad_compra = BUY_AMOUNT / precio
        
        # Promedio ponderado del precio de entrada
        total_coins = pf["coin"] + cantidad_compra
        total_cost = (pf["coin"] * pf["avg_price"]) + BUY_AMOUNT
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else precio
        
        pf["coin"] += cantidad_compra
        pf["usdt"] -= BUY_AMOUNT
        
        enviar_telegram(f"🔵 *{symbol} COMPRA PARCIAL*\nMonto: ${BUY_AMOUNT}\nPrecio: ${precio:,.4f}\nSaldo Crypto: {pf['coin']:.4f}\nSaldo USDT: ${pf['usdt']:.2f}")

    # --- LÓGICA DE VENTA (Total) ---
    # Vendemos si la señal es VENTA Y tenemos Crypto
    elif decision == "VENTA" and pf["coin"] * precio > 10: # Mínimo $10 para vender
        valor_venta = pf["coin"] * precio
        ganancia = valor_venta - (pf["coin"] * pf["avg_price"])
        
        pf["usdt"] += valor_venta
        pf["coin"] = 0
        pf["avg_price"] = 0
        pf["trades"] += 1
        
        icono = "✅" if ganancia > 0 else "❌"
        enviar_telegram(f"🟠 *{symbol} VENTA TOTAL*\nPrecio: ${precio:,.4f}\nResultado: {icono} ${ganancia:,.2f}\nSaldo USDT: ${pf['usdt']:.2f}")

    return {
        "symbol": symbol,
        "precio": precio,
        "score": score,
        "decision": decision,
        "detalles": razones,
        "grafico": generar_grafico(df, symbol),
        "portfolio": pf
    }

@app.get("/analisis")
def get_analisis(symbol: str = "BTCUSDT"):
    global market_data_cache
    if symbol not in SYMBOLS: symbol = "BTCUSDT"
    
    df = obtener_datos(symbol)
    if df.empty: return {"error": "Sin datos"}
    
    df = calcular_indicadores(df)
    resultado = ejecutar_estrategia(symbol, df)
    market_data_cache[symbol] = resultado
    
    try:
        tz = pytz.timezone('America/Argentina/Buenos_Aires')
        hora = datetime.now(tz).strftime("%H:%M:%S")
    except: hora = datetime.now().strftime("%H:%M:%S")
    
    resultado["update_time"] = hora
    return resultado

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)