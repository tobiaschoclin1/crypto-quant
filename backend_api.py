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
# Tu clave nueva
GEMINI_API_KEY = "AIzaSyBefrRTQIgNxgu0WU0vII2aAgk4EPxwvho" 
# --------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
INITIAL_CAPITAL = 1000.0
BUY_AMOUNT = 200.0 

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

        # Recuperar datos
        datos = market_data_cache.get(symbol, {})
        precio = datos.get("precio", 0)
        decision = datos.get("decision", "NEUTRAL")
        razones = datos.get("detalles", [])
        pf = datos.get("portfolio", {})
        
        # Datos extra para precisión
        soporte = datos.get("soporte", precio * 0.95)
        resistencia = datos.get("resistencia", precio * 1.05)

        contexto = f"""
        Actúa como un Trader Senior experto en {symbol}.
        
        DATOS DE MERCADO EN VIVO:
        - Precio Actual: ${precio:,.2f}
        - Señal del Sistema: {decision}
        - Soporte Inmediato (Piso): ${soporte:,.2f}
        - Resistencia (Techo): ${resistencia:,.2f}
        - Indicadores: {', '.join(razones)}
        
        MI PORTAFOLIO ({symbol}):
        - Dinero Disponible: ${pf.get('usdt', 0):,.2f}
        - Tenencia: {pf.get('coin', 0):,.4f} monedas (Precio Promedio: ${pf.get('avg_price', 0):,.2f})
        
        INSTRUCCIONES CLAVE:
        1. Si el usuario pregunta "a qué precio comprar/vender", NO digas "espera confirmación". 
           DALE EL NÚMERO DEL SOPORTE O RESISTENCIA como referencia concreta.
           Ejemplo: "Considera entrar cerca del soporte en ${soporte:,.2f}..."
        2. Responde corto, directo y útil.
        
        Usuario: "{user_message}"
        """
        
        payload = { "contents": [{ "parts": [{"text": contexto}] }] }
        headers = {"Content-Type": "application/json"}
        
        # 1. Autodescubrimiento (si hace falta)
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

        # 2. Chat
        url_chat = f"https://generativelanguage.googleapis.com/v1beta/models/{valid_model_name}:generateContent?key={api_key}"
        response = requests.post(url_chat, headers=headers, json=payload, timeout=8)
        
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
            r = requests.get(url, params=params, timeout=3)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=['t', 'o', 'h', 'l', 'c', 'v', 'x', 'x', 'x', 'x', 'x', 'x'])
                df['c'] = df['c'].astype(float)
                df['v'] = df['v'].astype(float)
                df['l'] = df['l'].astype(float) # Necesitamos Low para soporte
                df['h'] = df['h'].astype(float) # Necesitamos High para resistencia
                return df
        except: continue
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
    
    # CÁLCULO DE NIVELES CLAVE (Soporte/Resistencia de ultimas 50 velas)
    recent_data = df.tail(50)
    soporte_local = recent_data['l'].min()
    resistencia_local = recent_data['h'].max()
    
    tendencia_alcista = precio > current['ema200']
    cruce_macd_alcista = (prev['macd'] < prev['signal']) and (current['macd'] > current['signal'])
    
    score = 5
    razones = []
    
    if tendencia_alcista: score += 2; razones.append("Tendencia Alcista ✅")
    else: score -= 2; razones.append("Tendencia Bajista ⚠️")
    
    if current['rsi'] > 70: score -= 2; razones.append("Sobrecompra ⚠️")
    elif current['rsi'] < 30: score += 2; razones.append("Sobreventa 🚀")
    
    if cruce_macd_alcista: score += 3; razones.append("Cruce MACD Alcista 🔥")
    
    score = max(0, min(10, score))
    
    decision = "NEUTRAL"
    if tendencia_alcista and (cruce_macd_alcista or score >= 7): decision = "COMPRA"
    elif (not tendencia_alcista) or score <= 3: decision = "VENTA"
    
    pf = portfolios[symbol]
    
    if decision == "COMPRA" and pf["usdt"] >= BUY_AMOUNT:
        cantidad = BUY_AMOUNT / precio
        total_coins = pf["coin"] + cantidad
        total_cost = (pf["coin"] * pf["avg_price"]) + BUY_AMOUNT
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else precio
        pf["coin"] += cantidad
        pf["usdt"] -= BUY_AMOUNT
        enviar_telegram(f"🔵 *{symbol} COMPRA PARCIAL*\nPrecio: ${precio:,.2f}")

    elif decision == "VENTA" and pf["coin"] * precio > 10: 
        valor = pf["coin"] * precio
        ganancia = valor - (pf["coin"] * pf["avg_price"])
        pf["usdt"] += valor
        pf["coin"] = 0
        pf["trades"] += 1
        icono = "✅" if ganancia > 0 else "❌"
        enviar_telegram(f"🟠 *{symbol} VENTA TOTAL*\nResultado: {icono} ${ganancia:,.2f}")

    return {
        "symbol": symbol,
        "precio": precio,
        "score": score,
        "decision": decision,
        "detalles": razones,
        "soporte": soporte_local,      # DATO NUEVO
        "resistencia": resistencia_local, # DATO NUEVO
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
            "symbol": symbol,
            "precio": 0,
            "score": 0,
            "decision": "OFFLINE",
            "detalles": ["Error conexión Binance"],
            "grafico": "",
            "portfolio": cached,
            "update_time": datetime.now().strftime("%H:%M:%S")
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)