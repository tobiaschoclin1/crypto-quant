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
GEMINI_API_KEY = "PEGA_TU_CLAVE_AQUI" 
# --------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
GLOBAL_USDT = 0.0
real_portfolio = {
    sym: {"usdt": 0.0, "coin": 0.0, "avg_price": 0.0} 
    for sym in SYMBOLS
}
market_data_cache = {} 
valid_model_name = None
STOP_LOSS_PCT = 0.01 
TAKE_PROFIT_PCT = 0.015
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@app.post("/set_balance")
async def set_balance(request: Request):
    global GLOBAL_USDT, real_portfolio
    data = await request.json()
    GLOBAL_USDT = float(data.get("usdt", 0))
    for sym in SYMBOLS:
        real_portfolio[sym]["usdt"] = GLOBAL_USDT
        real_portfolio[sym]["coin"] = 0.0
        real_portfolio[sym]["avg_price"] = 0.0
    return {"status": "Turno iniciado", "usdt": GLOBAL_USDT}

@app.post("/registrar_trade")
async def registrar_trade(request: Request):
    global GLOBAL_USDT, real_portfolio
    data = await request.json()
    symbol = data.get("symbol")
    action = data.get("action")
    usdt_amount = float(data.get("usdt_amount", 0)) # AHORA ES EN DÓLARES
    price = float(data.get("price", 0))
    
    if symbol not in real_portfolio: return {"error": "Moneda no válida"}
    if price == 0: return {"error": "Precio inválido"}

    pf = real_portfolio[symbol]
    
    # Calculamos cuánta crypto es
    crypto_amount = usdt_amount / price
    
    if action == "COMPRA":
        if GLOBAL_USDT < usdt_amount: return {"error": "Saldo insuficiente"}
        
        total_coins = pf["coin"] + crypto_amount
        total_cost = (pf["coin"] * pf["avg_price"]) + usdt_amount
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else price
        pf["coin"] += crypto_amount
        GLOBAL_USDT -= usdt_amount
        
    elif action == "VENTA":
        # Para venta, usdt_amount es cuanto queremos recibir en dolares
        # Verificamos si tenemos esa cantidad en crypto
        if pf["coin"] * price < usdt_amount: return {"error": "No tienes suficiente crypto"}
        
        pf["coin"] -= crypto_amount
        if pf["coin"] < 0: pf["coin"] = 0
        GLOBAL_USDT += usdt_amount
        if pf["coin"] <= 0.000001: pf["avg_price"] = 0.0 # Reset si vendimos todo
        
    for s in SYMBOLS: real_portfolio[s]["usdt"] = GLOBAL_USDT
    return {"status": "OK", "nuevo_saldo": GLOBAL_USDT}

def obtener_datos(symbol):
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines"
    ]
    params = {"symbol": symbol, "interval": "1m", "limit": 100}
    for url in urls:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=3)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=['t', 'o', 'h', 'l', 'c', 'v', 'x', 'x', 'x', 'x', 'x', 'x'])
                for c in ['o', 'h', 'l', 'c', 'v']: df[c] = df[c].astype(float)
                return df
        except: continue
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
    if df.empty: 
        return {"error": True, "mensaje": "Sin conexión", "portfolio": real_portfolio[symbol]}
    
    df = calcular_indicadores(df)
    current = df.iloc[-1]
    precio = current['c']
    pf = real_portfolio[symbol]
    
    signal = "NEUTRAL"
    reasons = []
    
    if pf["coin"] > 0 and pf["avg_price"] > 0:
        pnl = (precio - pf["avg_price"]) / pf["avg_price"]
        if pnl <= -STOP_LOSS_PCT:
            signal = "VENTA FUERTE"
            reasons.append(f"🛑 STOP LOSS ({pnl*100:.2f}%)")
        elif pnl >= TAKE_PROFIT_PCT:
            signal = "VENTA FUERTE"
            reasons.append(f"💰 TAKE PROFIT ({pnl*100:.2f}%)")
            
    if signal == "NEUTRAL":
        if precio <= current['lower'] and current['rsi'] < 30:
            if pf["coin"] == 0:
                signal = "COMPRA"
                reasons.append("Soporte + RSI Bajo")
        elif (precio >= current['upper'] or current['rsi'] > 70):
            if pf["coin"] > 0:
                signal = "VENTA"
                reasons.append("Techo + RSI Alto")

    res = {
        "symbol": symbol, "precio": precio, "decision": signal, "detalles": reasons,
        "grafico": generar_grafico(df), "portfolio": pf, 
        "update_time": datetime.now().strftime("%H:%M:%S")
    }
    market_data_cache[symbol] = res
    return res

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
        Asesor Trading. {symbol}.
        Precio: ${datos.get('precio', 0):,.2f}. Señal: {datos.get('decision', 'NEUTRAL')}.
        Caja: ${GLOBAL_USDT:,.2f}. Tenencia: {pf.get('coin', 0):.4f}.
        Usuario: "{msg}". Responde corto.
        """
        payload = { "contents": [{ "parts": [{"text": contexto}] }] }
        headers = {"Content-Type": "application/json"}
        if not valid_model_name: valid_model_name = "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{valid_model_name}:generateContent?key={api_key}"
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200: return JSONResponse({"reply": r.json()['candidates'][0]['content']['parts'][0]['text']})
        return JSONResponse({"reply": "Error IA."})
    except: return JSONResponse({"reply": "Error interno."})

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return f.read()
    return "<h1>Error: No index.html</h1>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)