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
# Tu clave (que ya sabemos que funciona)
GEMINI_API_KEY = "AIzaSyAp1WURjJ03HhdB8NzkO1Rhre5-FqRtFIA" 
# --------------------

portfolio = {
    "usdt": 1000.0, "btc": 0.0, "in_market": False, 
    "entry_price": 0.0, "last_result": 0.0, "trades_count": 0
}
ultima_alerta = ""
ultimo_estado = {"decision": "NEUTRAL", "precio": 0, "score": 0, "razones": []}

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

# --- CHATBOT "HUMANIZADO" ---
@app.post("/chat")
async def chat_with_ai(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message", "")
        api_key = GEMINI_API_KEY.strip()

        # AQUÍ ESTÁ EL CAMBIO DE PERSONALIDAD
        contexto = f"""
        Actúa como un Analista de Trading experto pero con un tono humano, cercano y pedagógico.
        Evita el lenguaje excesivamente técnico o robótico. Si usas términos técnicos, explícalos brevemente.
        Usa analogías sencillas si es necesario.
        Tu objetivo es que el usuario entienda el "por qué" de la situación, no solo darle datos.
        Desarrolla tu respuesta en 2 o 3 oraciones claras.

        DATOS TÉCNICOS EN TIEMPO REAL:
        - Precio Bitcoin: ${ultimo_estado['precio']:,.2f}
        - Decisión del Algoritmo: {ultimo_estado['decision']}
        - Puntuación de Fuerza (Score): {ultimo_estado['score']}/10
        - Factores Clave: {', '.join(ultimo_estado['razones'])}
        
        Pregunta del Usuario: "{user_message}"
        """

        headers = {"Content-Type": "application/json"}
        payload = { "contents": [{ "parts": [{"text": contexto}] }] }

        # 1. Intentamos con el modelo estándar
        url_primaria = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        try:
            response = requests.post(url_primaria, headers=headers, json=payload, timeout=5)
            if response.status_code == 200:
                return JSONResponse({"reply": response.json()['candidates'][0]['content']['parts'][0]['text']})
        except:
            pass

        # 2. Respaldo (por si acaso)
        url_lista = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        resp_lista = requests.get(url_lista, timeout=5)
        
        if resp_lista.status_code == 200:
            data = resp_lista.json()
            if 'models' in data:
                for model_obj in data['models']:
                    nombre = model_obj['name'].replace('models/', '')
                    metodos = model_obj.get('supportedGenerationMethods', [])
                    if 'generateContent' in metodos:
                         url_reintento = f"https://generativelanguage.googleapis.com/v1beta/models/{nombre}:generateContent?key={api_key}"
                         resp_reintento = requests.post(url_reintento, headers=headers, json=payload, timeout=5)
                         if resp_reintento.status_code == 200:
                             return JSONResponse({"reply": resp_reintento.json()['candidates'][0]['content']['parts'][0]['text']})
        
        return JSONResponse({"reply": "Lo siento, mi cerebro de IA está teniendo un pequeño cortocircuito con Google. Intenta de nuevo."})

    except Exception as e:
        return JSONResponse({"reply": f"Error interno: {str(e)}"})

# --- FUNCIONES MATEMÁTICAS ---
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

def generar_grafico_base64(precios, ema_200):
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        data = precios[-100:] 
        ema_data = ema_200[-100:]
        x = range(len(data))
        ax.plot(x, data, color='#22d3ee', linewidth=2, label='Precio BTC')
        ax.plot(x, ema_data, color='#fbbf24', linewidth=1.5, linestyle='--', label='EMA 200')
        ax.fill_between(x, data, data.min(), color='#22d3ee', alpha=0.1)
        ax.axis('off'); ax.grid(False)
        ax.legend(loc='upper left', frameon=True, facecolor='#111827', edgecolor='none', labelcolor='white', fontsize=9)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except: return ""

def ejecutar_simulacion(precio_actual, score, ema_200_val):
    global portfolio
    tendencia_alcista_macro = precio_actual > ema_200_val
    
    if not portfolio["in_market"] and score >= 8:
        if tendencia_alcista_macro:
            portfolio["btc"] = portfolio["usdt"] / precio_actual
            portfolio["entry_price"] = precio_actual
            portfolio["usdt"] = 0
            portfolio["in_market"] = True
            enviar_telegram(f"🔵 *SIMULACIÓN: COMPRA*\nPrecio: ${precio_actual:,.2f}\nFiltro Tendencia: APROBADO ✅")

    elif portfolio["in_market"] and score <= 4:
        valor_venta = portfolio["btc"] * precio_actual
        costo_original = portfolio["btc"] * portfolio["entry_price"]
        ganancia_real = valor_venta - costo_original
        portfolio["usdt"] = valor_venta
        portfolio["btc"] = 0
        portfolio["in_market"] = False
        portfolio["last_result"] = ganancia_real
        portfolio["trades_count"] += 1
        icono = "✅" if ganancia_real > 0 else "❌"
        enviar_telegram(f"🟠 *SIMULACIÓN: VENTA*\nPrecio: ${precio_actual:,.2f}\nResultado: {icono} ${ganancia_real:,.2f}")

def calcular_todo():
    global ultima_alerta, ultimo_estado
    df = obtener_datos(limit=1000)
    if df.empty: return {"error": "Sin datos", "precio": 0, "score": 0, "decision": "Error"}

    precios = df['c'].values
    volumenes = df['v'].values
    precio_actual = precios[-1]
    
    ema_200 = df['c'].ewm(span=200, adjust=False).mean().values
    ema_200_actual = ema_200[-1]
    
    sma_100 = df["c"].rolling(window=100).mean().values
    sma_vol = df["v"].rolling(window=20).mean().values
    rsi = calcular_rsi(df["c"]).values
    suave = gaussian_filter1d(precios, sigma=1)
    vel = np.gradient(suave)

    score = 0
    razones = []

    if precio_actual > sma_100[-1]: score += 2; razones.append("Tendencia Corta Alcista ✅")
    else: razones.append("Tendencia Corta Bajista ❌")

    if precio_actual > ema_200_actual: score += 2; razones.append("Tendencia Macro Alcista ✅")
    else: score -= 2; razones.append("Tendencia Macro Bajista ⚠️")

    rsi_val = rsi[-1]
    if 30 <= rsi_val <= 70: score += 1; razones.append(f"RSI Neutro ({int(rsi_val)})")
    elif rsi_val > 70: score -= 2; razones.append(f"RSI Sobrecompra ({int(rsi_val)}) ⚠️")
    elif rsi_val < 30: score += 3; razones.append(f"RSI Sobreventa ({int(rsi_val)}) 🚀")

    if volumenes[-1] > sma_vol[-1]: score += 2; razones.append("Volumen Alto ✅")
    else: razones.append("Volumen Bajo ⚠️")

    if vel[-1] > 0: score += 1; razones.append("Impulso Positivo ✅")
    else: score -= 1; razones.append("Impulso Negativo ❌")

    score = max(0, min(10, score))

    if score >= 8: decision = "COMPRA FUERTE 🚀"
    elif score >= 6: decision = "COMPRA MODERADA 🟢"
    elif score >= 4: decision = "OBSERVAR 👀"
    elif score >= 2: decision = "VENTA 🔴"
    else: decision = "VENTA FUERTE 🩸"

    ejecutar_simulacion(precio_actual, score, ema_200_actual)

    ultimo_estado = {
        "decision": decision,
        "precio": precio_actual,
        "score": score,
        "razones": razones
    }

    clave_alerta = f"{decision}"
    if clave_alerta != ultima_alerta:
        if score >= 8 or score <= 3:
            enviar_telegram(f"📊 *ACTUALIZACIÓN*\nDecisión: {decision}\nPrecio: ${precio_actual:,.2f}\nScore: {score}/10")
            ultima_alerta = clave_alerta

    total_balance = portfolio["usdt"]
    if portfolio["in_market"]:
        total_balance = portfolio["btc"] * precio_actual

    try:
        tz_ba = pytz.timezone('America/Argentina/Buenos_Aires')
        hora_actual = datetime.now(tz_ba).strftime("%H:%M:%S")
    except:
        hora_actual = datetime.now().strftime("%H:%M:%S")

    return {
        "precio": precio_actual,
        "score": score,
        "decision": decision,
        "detalles": razones,
        "grafico_img": generar_grafico_base64(precios, ema_200),
        "update_time": hora_actual,
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