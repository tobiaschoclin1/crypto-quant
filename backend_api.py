from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import requests
import io
import base64
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
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

# --- MEMORIA DEL SISTEMA (Tu Libreta de Trabajo) ---
# Aquí guardamos lo que TÚ le dices que tienes.
real_portfolio = {
    sym: {"usdt": 0.0, "coin": 0.0, "avg_price": 0.0} 
    for sym in SYMBOLS
}
# Saldo global de USDT (compartido para todas las monedas o individual, 
# para simplificar lo haremos "Caja Única" visualmente en el frontend, 
# pero internamente trackeamos disponibilidad).
GLOBAL_USDT = 0.0

market_data_cache = {} 
valid_model_name = None

# --- CONFIGURACIÓN ESTRATEGIA (Scalping Sniper) ---
STOP_LOSS_PCT = 0.01  # 1% Riesgo
TAKE_PROFIT_PCT = 0.015 # 1.5% Ganancia

# --- ENDPOINTS DE GESTIÓN (TU OFICINA) ---

@app.post("/set_balance")
async def set_balance(request: Request):
    global GLOBAL_USDT, real_portfolio
    data = await request.json()
    
    # Reiniciamos el día
    GLOBAL_USDT = float(data.get("usdt", 0))
    
    # Opcional: Si ya tienes criptos compradas de antes
    # Esperamos un dict: {"BTCUSDT": 0.5, "ETHUSDT": 2.0}
    holdings = data.get("holdings", {})
    
    for sym in SYMBOLS:
        qty = float(holdings.get(sym, 0))
        real_portfolio[sym] = {
            "usdt": GLOBAL_USDT, # Referencia
            "coin": qty,
            "avg_price": 0.0 # Si traes de antes, asumimos 0 o habría que pedir precio
        }
        
    return {"status": "Turno iniciado", "usdt": GLOBAL_USDT}

@app.post("/registrar_trade")
async def registrar_trade(request: Request):
    global GLOBAL_USDT, real_portfolio
    data = await request.json()
    
    symbol = data.get("symbol")
    action = data.get("action") # "COMPRA" o "VENTA"
    amount_crypto = float(data.get("amount", 0)) # Cuanta cripto compraste/vendiste
    price = float(data.get("price", 0)) # A qué precio
    
    if symbol not in real_portfolio: return {"error": "Moneda no válida"}
    
    pf = real_portfolio[symbol]
    total_usd = amount_crypto * price
    
    if action == "COMPRA":
        # Actualizamos promedio de compra
        total_coins = pf["coin"] + amount_crypto
        total_cost = (pf["coin"] * pf["avg_price"]) + total_usd
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else price
        
        pf["coin"] += amount_crypto
        GLOBAL_USDT -= total_usd
        
    elif action == "VENTA":
        pf["coin"] -= amount_crypto
        if pf["coin"] < 0: pf["coin"] = 0 # Evitar negativos
        GLOBAL_USDT += total_usd
        # Si vendemos todo, reseteamos precio promedio
        if pf["coin"] == 0: pf["avg_price"] = 0.0
        
    # Sincronizar USDT en todos los pares (Concepto Caja Única)
    for s in SYMBOLS: real_portfolio[s]["usdt"] = GLOBAL_USDT
        
    return {"status": "Operación registrada", "nuevo_saldo": GLOBAL_USDT, "tenencia": pf["coin"]}


# --- LÓGICA DE TRADING (El Analista) ---
# (Usamos la lógica Sniper/Scalper del nivel anterior, pero SIN ejecutar compras automáticas)

def obtener_datos(symbol):
    # Usamos 1m para rapidez
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1m", "limit": 100}
        r = requests.get(url, params=params, timeout=2)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=['t', 'o', 'h', 'l', 'c', 'v', 'x', 'x', 'x', 'x', 'x', 'x'])
            for c in ['o', 'h', 'l', 'c', 'v']: df[c] = df[c].astype(float)
            return df
    except: pass
    return pd.DataFrame()

def calcular_indicadores(df):
    df['sma20'] = df['c'].rolling(window=20).mean()
    df['std20'] = df['c'].rolling(window=20).std()
    df['upper'] = df['sma20'] + (df['std20'] * 2)
    df['lower'] = df['sma20'] - (df['std20'] * 2)
    
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def generar_grafico(df):
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        data = df['c'].tail(60)
        upper = df['upper'].tail(60)
        lower = df['lower'].tail(60)
        x = range(len(data))
        ax.plot(x, data, color='#22d3ee', linewidth=2)
        ax.plot(x, upper, color='#a78bfa', alpha=0.5)
        ax.plot(x, lower, color='#a78bfa', alpha=0.5)
        ax.fill_between(x, upper, lower, color='#a78bfa', alpha=0.05)
        ax.axis('off'); ax.grid(False)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except: return ""

@app.get("/analisis")
def get_analisis(symbol: str = "BTCUSDT"):
    global market_data_cache, real_portfolio
    if symbol not in SYMBOLS: symbol = "BTCUSDT"
    
    df = obtener_datos(symbol)
    if df.empty: return {"error": "Offline", "portfolio": real_portfolio[symbol]}
    
    df = calcular_indicadores(df)
    current = df.iloc[-1]
    precio = current['c']
    pf = real_portfolio[symbol]
    
    # --- ESTRATEGIA (CONSEJERO) ---
    signal = "NEUTRAL"
    reasons = []
    
    # 1. Chequeo de Posición (Stop Loss / Take Profit)
    if pf["coin"] > 0 and pf["avg_price"] > 0:
        pnl = (precio - pf["avg_price"]) / pf["avg_price"]
        if pnl <= -STOP_LOSS_PCT:
            signal = "VENTA FUERTE"
            reasons.append(f"🛑 STOP LOSS ACTIVADO ({pnl*100:.2f}%)")
        elif pnl >= TAKE_PROFIT_PCT:
            signal = "VENTA FUERTE"
            reasons.append(f"💰 TAKE PROFIT ACTIVADO ({pnl*100:.2f}%)")
            
    # 2. Análisis Técnico (Solo si no hay señal de salida forzada)
    if signal == "NEUTRAL":
        # Compra: Precio bajo banda inferior + RSI bajo
        if precio <= current['lower'] and current['rsi'] < 30:
            if pf["coin"] == 0: # Solo sugerir compra si no tenemos
                signal = "COMPRA"
                reasons.append("Rebote en Soporte + RSI Bajo")
        
        # Venta: Precio alto banda superior + RSI alto
        elif (precio >= current['upper'] or current['rsi'] > 70):
            if pf["coin"] > 0:
                signal = "VENTA"
                reasons.append("Techo alcanzado + RSI Alto")

    res = {
        "symbol": symbol,
        "precio": precio,
        "decision": signal,
        "detalles": reasons,
        "grafico": generar_grafico(df),
        "portfolio": pf, # Enviamos el portafolio REAL
        "update_time": datetime.now().strftime("%H:%M:%S")
    }
    market_data_cache[symbol] = res
    return res

# --- CHATBOT (Consciente de tu saldo real) ---
@app.post("/chat")
async def chat_with_ai(request: Request):
    global valid_model_name, GLOBAL_USDT
    try:
        body = await request.json()
        msg = body.get("message", "")
        symbol = body.get("symbol", "BTCUSDT")
        api_key = GEMINI_API_KEY.strip()
        
        datos = market_data_cache.get(symbol, {})
        pf = real_portfolio.get(symbol, {})
        
        contexto = f"""
        Eres un Asesor de Trading de Oficina.
        Tu jefe (el usuario) está operando {symbol}.
        
        ESTADO DEL MERCADO:
        - Precio: ${datos.get('precio', 0):,.2f}
        - Recomendación Sistema: {datos.get('decision', 'NEUTRAL')}
        
        ESTADO DE CUENTA (REAL):
        - Caja USDT: ${GLOBAL_USDT:,.2f}
        - Tenencia {symbol}: {pf.get('coin', 0):.4f} (Precio entrada: ${pf.get('avg_price', 0):,.2f})
        
        Usuario dice: "{msg}"
        Responde profesional, corto y directo. Ayúdale a ganar dinero.
        """
        
        payload = { "contents": [{ "parts": [{"text": contexto}] }] }
        headers = {"Content-Type": "application/json"}
        
        if not valid_model_name: valid_model_name = "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{valid_model_name}:generateContent?key={api_key}"
        
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200: return JSONResponse({"reply": r.json()['candidates'][0]['content']['parts'][0]['text']})
        return JSONResponse({"reply": "Error de conexión IA."})
        
    except: return JSONResponse({"reply": "Error interno."})

# --- CORRECCIÓN AQUÍ ---
@app.get("/", response_class=HTMLResponse) # <--- Agregamos response_class=HTMLResponse
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: No index.html</h1>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)