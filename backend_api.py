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
GEMINI_API_KEY = "AIzaSyBmeV-fa7Buf2EKoVzRSm-PF6R8tJF2E9c" 
# --------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
INITIAL_CAPITAL = 1000.0
BUY_AMOUNT = 200.0 

# CONFIGURACIÓN SCALPING (Muy Agresiva)
STOP_LOSS_PCT = 0.003  # 0.3% (Cortar rapidísimo)
TAKE_PROFIT_PCT = 0.008 # 0.8% (Asegurar ganancia rápida)

portfolios = {
    sym: {"usdt": INITIAL_CAPITAL, "coin": 0.0, "avg_price": 0.0, "trades": 0}
    for sym in SYMBOLS
}
market_data_cache = {} 
valid_model_name = None

def enviar_telegram(mensaje):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=2)
    except: pass

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: No se encuentra index.html</h1>"

@app.post("/chat")
async def chat_with_ai(request: Request):
    global valid_model_name
    try:
        body = await request.json()
        user_message = body.get("message", "")
        symbol = body.get("symbol", "BTCUSDT")
        api_key = GEMINI_API_KEY.strip()

        datos = market_data_cache.get(symbol, {})
        precio = datos.get("precio", 0)
        decision = datos.get("decision", "NEUTRAL")
        razones = datos.get("detalles", [])
        soporte = datos.get("soporte", 0)
        resistencia = datos.get("resistencia", 0)

        contexto = f"""
        Actúa como un Scalper Trader agresivo experto en {symbol}.
        MERCADO (1min): Precio ${precio:,.2f} | Señal: {decision}
        Soporte/Resistencia: ${soporte:,.2f} / ${resistencia:,.2f}
        Razones: {', '.join(razones)}
        Usuario: "{user_message}"
        Responde en 1 frase ultra corta y directa.
        """
        
        payload = { "contents": [{ "parts": [{"text": contexto}] }] }
        headers = {"Content-Type": "application/json"}
        
        if not valid_model_name:
            valid_model_name = "gemini-1.5-flash"

        url_chat = f"https://generativelanguage.googleapis.com/v1beta/models/{valid_model_name}:generateContent?key={api_key}"
        response = requests.post(url_chat, headers=headers, json=payload, timeout=10) # Timeout corto para velocidad
        
        if response.status_code == 200:
            return JSONResponse({"reply": response.json()['candidates'][0]['content']['parts'][0]['text']})
        elif response.status_code == 404:
             valid_model_name = "gemini-1.0-pro" # Fallback rápido
             return JSONResponse({"reply": "Recalibrando IA..."})
        else:
            return JSONResponse({"reply": "Silencio de radio (Error IA)."})

    except Exception as e:
        return JSONResponse({"reply": "Error sistema."})

# --- LÓGICA DE MERCADO ---
def obtener_datos(symbol):
    # Usamos 1m (1 minuto) para Scalping puro
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines"
    ]
    params = {"symbol": symbol, "interval": "1m", "limit": 100} # Solo necesitamos 100 velas
    
    for url in urls:
        try:
            r = requests.get(url, params=params, timeout=2) # Timeout muy bajo para no colgarse
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=['t', 'o', 'h', 'l', 'c', 'v', 'x', 'x', 'x', 'x', 'x', 'x'])
                for c in ['o', 'h', 'l', 'c', 'v']: df[c] = df[c].astype(float)
                return df
        except: continue
    return pd.DataFrame()

def calcular_indicadores(df):
    # Estrategia Scalping: Bollinger + RSI Rápido
    df['sma20'] = df['c'].rolling(window=20).mean()
    df['std20'] = df['c'].rolling(window=20).std()
    df['upper'] = df['sma20'] + (df['std20'] * 2)
    df['lower'] = df['sma20'] - (df['std20'] * 2)
    
    # RSI de 7 periodos (más sensible que el de 14)
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def generar_grafico(df, symbol):
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        data = df['c'].tail(60) # Últimos 60 minutos
        upper = df['upper'].tail(60)
        lower = df['lower'].tail(60)
        x = range(len(data))
        
        ax.plot(x, data, color='#22d3ee', linewidth=2, label='Precio (1m)')
        ax.plot(x, upper, color='#a78bfa', linewidth=0.5, alpha=0.5)
        ax.plot(x, lower, color='#a78bfa', linewidth=0.5, alpha=0.5)
        ax.fill_between(x, upper, lower, color='#a78bfa', alpha=0.05)
        
        ax.axis('off'); ax.grid(False)
        ax.legend(loc='upper left', frameon=False, fontsize=8)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except: return ""

def ejecutar_estrategia(symbol, df):
    global portfolios
    
    current = df.iloc[-1]
    precio = current['c']
    pf = portfolios[symbol]
    
    # GESTIÓN RIESGO MILIMÉTRICA
    stop_triggered = False
    take_profit_triggered = False
    
    if pf["coin"] > 0:
        pnl_pct = (precio - pf["avg_price"]) / pf["avg_price"]
        if pnl_pct <= -STOP_LOSS_PCT: stop_triggered = True
        elif pnl_pct >= TAKE_PROFIT_PCT: take_profit_triggered = True

    # SEÑALES DE SCALPING
    # Compra: Precio toca banda inferior Y RSI < 30 (Rebote inminente)
    signal_buy = (precio <= current['lower']) and (current['rsi'] < 30)
    
    # Venta: Precio toca banda superior O RSI > 70 (Agotamiento)
    signal_sell = (precio >= current['upper']) or (current['rsi'] > 70)

    decision = "NEUTRAL"
    razones = []

    if stop_triggered: decision = "VENTA"; razones.append("STOP LOSS (Scalp) 🛑")
    elif take_profit_triggered: decision = "VENTA"; razones.append("TAKE PROFIT (Scalp) 💰")
    elif signal_buy: decision = "COMPRA"; razones.append("Rebote Banda Inferior 💎")
    elif signal_sell and pf["coin"] > 0: decision = "VENTA"; razones.append("Techo Banda Superior 📉")

    score = 5
    if decision == "COMPRA": score = 9
    elif decision == "VENTA": score = 2

    # EJECUCIÓN
    if decision == "COMPRA" and pf["usdt"] >= BUY_AMOUNT:
        cantidad = BUY_AMOUNT / precio
        total_coins = pf["coin"] + cantidad
        total_cost = (pf["coin"] * pf["avg_price"]) + BUY_AMOUNT
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else precio
        pf["coin"] += cantidad
        pf["usdt"] -= BUY_AMOUNT
        enviar_telegram(f"⚡ *SCALPING {symbol} COMPRA*\nPrecio: ${precio:,.2f}")

    elif decision == "VENTA" and pf["coin"] * precio > 5: 
        valor = pf["coin"] * precio
        ganancia = valor - (pf["coin"] * pf["avg_price"])
        pf["usdt"] += valor
        pf["coin"] = 0
        pf["avg_price"] = 0
        pf["trades"] += 1
        icono = "✅" if ganancia > 0 else "❌"
        enviar_telegram(f"⚡ *SCALPING {symbol} VENTA*\nResultado: {icono} ${ganancia:,.2f}")

    return {
        "symbol": symbol,
        "precio": precio,
        "score": score,
        "decision": decision,
        "detalles": razones,
        "soporte": current['lower'],
        "resistencia": current['upper'],
        "grafico": generar_grafico(df, symbol),
        "portfolio": pf
    }

@app.get("/analisis")
def get_analisis(symbol: str = "BTCUSDT"):
    global market_data_cache
    if symbol not in SYMBOLS: symbol = "BTCUSDT"
    
    df = obtener_datos(symbol)
    if df.empty:
        # Modo Fallback para no romper el frontend
        cached = portfolios[symbol]
        return {
            "symbol": symbol, "precio": 0, "decision": "OFFLINE", 
            "portfolio": cached, "update_time": "Reintentando..."
        }
    
    df = calcular_indicadores(df)
    resultado = ejecutar_estrategia(symbol, df)
    market_data_cache[symbol] = resultado
    
    try:
        tz = pytz.timezone('America/Argentina/Buenos_Aires')
        resultado["update_time"] = datetime.now(tz).strftime("%H:%M:%S")
    except: 
        resultado["update_time"] = datetime.now().strftime("%H:%M:%S")
        
    return resultado

@app.get("/update_all")
def update_all_symbols():
    resumen = []
    for sym in SYMBOLS:
        get_analisis(sym)
        resumen.append(sym)
    return {"status": "Scalping Cycle Done", "checked": resumen}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)