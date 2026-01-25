from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import requests
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
GEMINI_API_KEY = "PEGA_TU_CLAVE_AQUI" 
# --------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
INITIAL_CAPITAL = 1000.0
BUY_AMOUNT = 200.0 

# CONFIGURACIÓN SNIPER SCALPING
STOP_LOSS_PCT = 0.005  # 0.5% (Un poco más de aire)
TAKE_PROFIT_PCT = 0.012 # 1.2% (Buscamos recorridos más seguros)

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
        tendencia_macro = datos.get("tendencia_macro", "NEUTRAL")

        contexto = f"""
        Actúa como un Trader Sniper experto en {symbol}.
        MERCADO: Precio ${precio:,.2f} | Señal: {decision}
        TENDENCIA MACRO (1H): {tendencia_macro}
        Razones: {', '.join(razones)}
        Usuario: "{user_message}"
        Responde en 1 frase táctica.
        """
        
        payload = { "contents": [{ "parts": [{"text": contexto}] }] }
        headers = {"Content-Type": "application/json"}
        
        if not valid_model_name: valid_model_name = "gemini-1.5-flash"
        url_chat = f"https://generativelanguage.googleapis.com/v1beta/models/{valid_model_name}:generateContent?key={api_key}"
        
        try:
            response = requests.post(url_chat, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                return JSONResponse({"reply": response.json()['candidates'][0]['content']['parts'][0]['text']})
        except: pass
        return JSONResponse({"reply": "Analizando fractales..."})

    except Exception as e:
        return JSONResponse({"reply": "Error sistema."})

# --- LÓGICA DE MERCADO ---
def obtener_datos(symbol, interval, limit=100):
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines"
    ]
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for url in urls:
        try:
            r = requests.get(url, params=params, timeout=2)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=['t', 'o', 'h', 'l', 'c', 'v', 'x', 'x', 'x', 'x', 'x', 'x'])
                for c in ['o', 'h', 'l', 'c', 'v']: df[c] = df[c].astype(float)
                return df
        except: continue
    return pd.DataFrame()

def calcular_adx(df, period=14):
    df['tr1'] = df['h'] - df['l']
    df['tr2'] = abs(df['h'] - df['c'].shift(1))
    df['tr3'] = abs(df['l'] - df['c'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    df['up_move'] = df['h'] - df['h'].shift(1)
    df['down_move'] = df['l'].shift(1) - df['l']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
    
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()
    return df

def calcular_indicadores(df):
    # Bollinger
    df['sma20'] = df['c'].rolling(window=20).mean()
    df['std20'] = df['c'].rolling(window=20).std()
    df['upper'] = df['sma20'] + (df['std20'] * 2)
    df['lower'] = df['sma20'] - (df['std20'] * 2)
    
    # RSI
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ADX (Fuerza de tendencia)
    df = calcular_adx(df)
    
    return df

def generar_grafico(df, symbol):
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        data = df['c'].tail(60)
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

def ejecutar_estrategia(symbol):
    global portfolios
    
    # 1. OBTENER DATOS MICRO (1 Minuto) - Para entrar
    df_micro = obtener_datos(symbol, "1m", 100)
    if df_micro.empty: return None
    df_micro = calcular_indicadores(df_micro)
    
    # 2. OBTENER DATOS MACRO (1 Hora) - Para filtrar
    df_macro = obtener_datos(symbol, "1h", 50)
    tendencia_macro = "NEUTRAL"
    if not df_macro.empty:
        ema50_macro = df_macro['c'].ewm(span=50, adjust=False).mean().iloc[-1]
        precio_macro = df_macro['c'].iloc[-1]
        tendencia_macro = "ALCISTA" if precio_macro > ema50_macro else "BAJISTA"

    current = df_micro.iloc[-1]
    precio = current['c']
    pf = portfolios[symbol]
    
    # GESTIÓN RIESGO
    stop_triggered = False
    take_profit_triggered = False
    if pf["coin"] > 0:
        pnl_pct = (precio - pf["avg_price"]) / pf["avg_price"]
        if pnl_pct <= -STOP_LOSS_PCT: stop_triggered = True
        elif pnl_pct >= TAKE_PROFIT_PCT: take_profit_triggered = True

    # SEÑALES (Confirmadas con Macro)
    # Solo compramos si MACRO es ALCISTA y el ADX en micro tiene fuerza (>20)
    adx_fuerte = current['adx'] > 20
    signal_buy = (precio <= current['lower']) and (current['rsi'] < 30)
    signal_sell = (precio >= current['upper']) or (current['rsi'] > 70)

    decision = "NEUTRAL"
    razones = []
    
    razones.append(f"Macro (1H): {tendencia_macro}")

    if stop_triggered: decision = "VENTA"; razones.append("STOP LOSS 🛑")
    elif take_profit_triggered: decision = "VENTA"; razones.append("TAKE PROFIT 💰")
    
    elif signal_buy:
        if tendencia_macro == "ALCISTA" and adx_fuerte:
            decision = "COMPRA"
            razones.append("Entrada Sniper (Trend + Rebote) 🎯")
        elif tendencia_macro == "BAJISTA":
            razones.append("Compra anulada por Tendencia Bajista ⚠️")
        elif not adx_fuerte:
            razones.append("Mercado lateral (ADX bajo) 💤")
            
    elif signal_sell and pf["coin"] > 0: 
        decision = "VENTA"
        razones.append("Salida Técnica 📉")

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
        enviar_telegram(f"🎯 *SNIPER {symbol} COMPRA*\nPrecio: ${precio:,.2f}")

    elif decision == "VENTA" and pf["coin"] * precio > 5: 
        valor = pf["coin"] * precio
        ganancia = valor - (pf["coin"] * pf["avg_price"])
        pf["usdt"] += valor
        pf["coin"] = 0
        pf["avg_price"] = 0
        pf["trades"] += 1
        icono = "✅" if ganancia > 0 else "❌"
        enviar_telegram(f"🎯 *SNIPER {symbol} VENTA*\nResultado: {icono} ${ganancia:,.2f}")

    return {
        "symbol": symbol, "precio": precio, "score": score, "decision": decision,
        "detalles": razones, "tendencia_macro": tendencia_macro,
        "grafico": generar_grafico(df_micro, symbol), "portfolio": pf
    }

@app.get("/analisis")
def get_analisis(symbol: str = "BTCUSDT"):
    global market_data_cache
    if symbol not in SYMBOLS: symbol = "BTCUSDT"
    
    resultado = ejecutar_estrategia(symbol)
    
    if not resultado:
        cached = portfolios[symbol]
        return {"symbol": symbol, "precio": 0, "decision": "OFFLINE", "portfolio": cached}
    
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
    return {"status": "Sniper Cycle Done", "checked": resumen}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)