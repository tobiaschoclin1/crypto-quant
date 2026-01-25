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
# ⚠️ TU CLAVE NUEVA ⚠️
GEMINI_API_KEY = "PEGA_TU_CLAVE_AQUI" 
# --------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
INITIAL_CAPITAL = 1000.0
BUY_AMOUNT = 200.0 

# GESTIÓN DE RIESGO
STOP_LOSS_PCT = 0.02  # Cortar pérdidas al 2%
TAKE_PROFIT_PCT = 0.06 # Buscar ganancias del 6%

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
        pf = datos.get("portfolio", {})
        soporte = datos.get("soporte", 0)
        resistencia = datos.get("resistencia", 0)

        contexto = f"""
        Actúa como un Trader Senior experto en {symbol}.
        DATOS DE MERCADO:
        - Precio: ${precio:,.2f}
        - Señal: {decision}
        - Soportes/Resistencias: ${soporte:,.2f} / ${resistencia:,.2f}
        - Razones Técnicas: {', '.join(razones)}
        PORTAFOLIO:
        - USDT: ${pf.get('usdt', 0):,.2f}
        - Crypto: {pf.get('coin', 0):,.4f}
        Usuario: "{user_message}"
        Responde corto y directo.
        """
        
        payload = { "contents": [{ "parts": [{"text": contexto}] }] }
        headers = {"Content-Type": "application/json"}
        
        if not valid_model_name:
            try:
                resp_list = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", timeout=5)
                if resp_list.status_code == 200:
                    for model in resp_list.json().get('models', []):
                        if 'generateContent' in model.get('supportedGenerationMethods', []):
                            valid_model_name = model['name'].replace("models/", "")
                            break
            except: pass
        if not valid_model_name: valid_model_name = "gemini-1.5-flash"

        url_chat = f"https://generativelanguage.googleapis.com/v1beta/models/{valid_model_name}:generateContent?key={api_key}"
        response = requests.post(url_chat, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return JSONResponse({"reply": response.json()['candidates'][0]['content']['parts'][0]['text']})
        elif response.status_code == 404:
             valid_model_name = None 
             return JSONResponse({"reply": "Modelo reiniciando... pregunta de nuevo."})
        else:
            return JSONResponse({"reply": f"Error IA: {response.text}"})

    except Exception as e:
        return JSONResponse({"reply": f"Error interno: {str(e)}"})

# --- LÓGICA DE MERCADO ---
def obtener_datos(symbol):
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines"
    ]
    params = {"symbol": symbol, "interval": "15m", "limit": 200}
    for url in urls:
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=['t', 'o', 'h', 'l', 'c', 'v', 'x', 'x', 'x', 'x', 'x', 'x'])
                cols = ['o', 'h', 'l', 'c', 'v']
                for c in cols: df[c] = df[c].astype(float)
                return df
        except: continue
    return pd.DataFrame()

def calcular_indicadores(df):
    # RSI
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMA 200 (Tendencia Macro)
    df['ema200'] = df['c'].ewm(span=200, adjust=False).mean()
    
    # MACD
    ema12 = df['c'].ewm(span=12, adjust=False).mean()
    ema26 = df['c'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # BANDAS DE BOLLINGER
    df['sma20'] = df['c'].rolling(window=20).mean()
    df['std20'] = df['c'].rolling(window=20).std()
    df['bollinger_upper'] = df['sma20'] + (df['std20'] * 2)
    df['bollinger_lower'] = df['sma20'] - (df['std20'] * 2)
    
    return df

def ejecutar_estrategia(symbol, df):
    global portfolios
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    precio = current['c']
    pf = portfolios[symbol]
    
    # 1. ANÁLISIS DE TENDENCIA (LA CLAVE ANTI-CAÍDAS)
    # Calculamos la pendiente de la EMA 200 (Si baja, es mercado bajista, NO COMPRAR)
    ema_actual = current['ema200']
    ema_prev = df.iloc[-5]['ema200'] # Hace 5 velas
    pendiente_bajista = ema_actual < ema_prev # La línea va hacia abajo
    
    # Precio debajo de EMA 200 es zona de peligro
    zona_peligro = precio < ema_actual

    # 2. GESTIÓN DE RIESGO ACTIVA
    stop_triggered = False
    take_profit_triggered = False
    
    if pf["coin"] > 0:
        pnl_pct = (precio - pf["avg_price"]) / pf["avg_price"]
        if pnl_pct <= -STOP_LOSS_PCT: stop_triggered = True
        elif pnl_pct >= TAKE_PROFIT_PCT: take_profit_triggered = True

    # 3. SEÑALES TÉCNICAS
    cruce_macd_alcista = (prev['macd'] < prev['signal']) and (current['macd'] > current['signal'])
    # RSI Muy bajo (Pánico real)
    rsi_panico = current['rsi'] < 25 
    
    razones = []
    score = 5

    # Lógica Defensiva
    if pendiente_bajista: 
        score = 0
        razones.append("Tendencia Bajista Fuerte (EMA 📉)")
    elif zona_peligro:
        score -= 2
        razones.append("Precio bajo EMA 200 ⚠️")
    else:
        score += 2
        razones.append("Tendencia Saludable ✅")

    if cruce_macd_alcista: score += 2; razones.append("Cruce MACD 🟢")
    if rsi_panico: score += 2; razones.append("Oportunidad Rebote (RSI bajo) 💎")

    # DECISIÓN FINAL
    decision = "NEUTRAL"

    # Prioridad 1: Salvar capital (Stop Loss / Take Profit)
    if stop_triggered or take_profit_triggered:
        decision = "VENTA"
    
    # Prioridad 2: Compra SEGURA
    # SOLO compramos si la tendencia NO es bajista fuerte
    # O si hay un rebote técnico muy claro (MACD + RSI Pánico) aunque estemos abajo
    elif (not pendiente_bajista) and (score >= 7 or (cruce_macd_alcista and rsi_panico)):
        decision = "COMPRA"
    
    # Prioridad 3: Venta técnica
    elif pendiente_bajista and pf["coin"] > 0:
        decision = "VENTA" # Salir si la tendencia se rompe
    
    # --- EJECUCIÓN ---
    if decision == "COMPRA" and pf["usdt"] >= BUY_AMOUNT:
        cantidad = BUY_AMOUNT / precio
        total_coins = pf["coin"] + cantidad
        total_cost = (pf["coin"] * pf["avg_price"]) + BUY_AMOUNT
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else precio
        pf["coin"] += cantidad
        pf["usdt"] -= BUY_AMOUNT
        enviar_telegram(f"🔵 *{symbol} COMPRA*\nPrecio: ${precio:,.2f}\nRazones: {', '.join(razones)}")

    elif decision == "VENTA" and pf["coin"] * precio > 10: 
        valor = pf["coin"] * precio
        ganancia = valor - (pf["coin"] * pf["avg_price"])
        
        motivo = "Técnica"
        if stop_triggered: motivo = "STOP LOSS 🛑"
        if take_profit_triggered: motivo = "TAKE PROFIT 💰"
        if pendiente_bajista: motivo = "TENDENCIA ROTA 📉"

        pf["usdt"] += valor
        pf["coin"] = 0
        pf["avg_price"] = 0
        pf["trades"] += 1
        
        icono = "✅" if ganancia > 0 else "❌"
        enviar_telegram(f"🟠 *{symbol} VENTA ({motivo})*\nResultado: {icono} ${ganancia:,.2f}")

    return {
        "symbol": symbol,
        "precio": precio,
        "score": score,
        "decision": decision,
        "detalles": razones,
        "soporte": df.tail(50)['l'].min(),
        "resistencia": df.tail(50)['h'].max(),
        "grafico": generar_grafico(df, symbol),
        "portfolio": pf
    }

@app.get("/analisis")
def get_analisis(symbol: str = "BTCUSDT"):
    global market_data_cache
    if symbol not in SYMBOLS: symbol = "BTCUSDT"
    
    df = obtener_datos(symbol)
    if df.empty:
        cached = portfolios[symbol]
        return {
            "symbol": symbol, "precio": 0, "decision": "OFFLINE", "portfolio": cached, "update_time": "00:00:00"
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

# ENDPOINT UPTIMEROBOT
@app.get("/update_all")
def update_all_symbols():
    resumen = []
    for sym in SYMBOLS:
        dato = get_analisis(sym)
        resumen.append({"moneda": sym, "accion": dato["decision"]})
    return {"status": "OK", "data": resumen}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)