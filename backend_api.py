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
import google.generativeai as genai # <--- NUEVA LIBRERÍA

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CREDENCIALES (PEGA TUS DATOS AQUÍ) ---
TELEGRAM_TOKEN = "8352173352:AAF1EuGRmTdbyDD_edQodfp3UPPeTWqqgwA" 
TELEGRAM_CHAT_ID = "793016927"
GEMINI_API_KEY = "AIzaSyADQkZ4LuS7T_smMl1kFIVSZ07lU7bp7iU" 
# -------------------------------------------

# Configurar Gemini
genai.configure(api_key=GEMINI_API_KEY)

portfolio = {
    "usdt": 1000.0, "btc": 0.0, "in_market": False, 
    "entry_price": 0.0, "last_result": 0.0, "trades_count": 0
}
ultima_alerta = ""
# Guardamos el último análisis para dárselo a Gemini
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

# --- NUEVO: ENDPOINT DE CHAT ---
@app.post("/chat")
async def chat_with_ai(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message", "")
        
        # Construimos el contexto con los datos reales del bot
        contexto = f"""
        Actúa como un Asesor de Trading Cuantitativo Senior.
        DATOS EN TIEMPO REAL DEL MERCADO (BTC/USDT):
        - Precio: ${ultimo_estado['precio']:,.2f}
        - Decisión del Algoritmo: {ultimo_estado['decision']}
        - Score Técnico: {ultimo_estado['score']}/10
        - Indicadores Clave: {', '.join(ultimo_estado['razones'])}
        - Portafolio Simulado: {'INVERTIDO' if portfolio['in_market'] else 'LÍQUIDO (USDT)'}
        
        El usuario pregunta: "{user_message}"
        
        Responde corto, directo y sarcástico si el usuario pregunta algo obvio. 
        Usa emojis. Basa tu respuesta estrictamente en los datos provistos.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(contexto)
        
        return JSONResponse({"reply": response.text})
    except Exception as e:
        return JSONResponse({"reply": f"Error cerebral: {str(e)}"})

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
    if df.empty: return {"error": "Sin datos"}

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

    # Actualizar estado global para Gemini
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

    return {
        "precio": precio_actual,
        "score": score,
        "decision": decision,
        "detalles": razones,
        "grafico_img": generar_grafico_base64(precios, ema_200),
        "update_time": datetime.now().strftime("%H:%M:%S"),
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