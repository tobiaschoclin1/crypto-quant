from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import uvicorn
import pandas as pd
import os
from datetime import datetime
import pytz 
import yfinance as yf 

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CREDENCIALES ---
TELEGRAM_TOKEN = "8352173352:AAF1EuGRmTdbyDD_edQodfp3UPPeTWqqgwA" 
TELEGRAM_CHAT_ID = "793016927"
# --------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
GLOBAL_USDT = 0.0 
real_portfolio = {
    sym: {"usdt": 0.0, "coin": 0.0, "avg_price": 0.0} 
    for sym in SYMBOLS
}
market_data_cache = {} 
STOP_LOSS_PCT = 0.01 
TAKE_PROFIT_PCT = 0.015

@app.post("/set_balance")
async def set_balance(request: Request):
    global GLOBAL_USDT, real_portfolio
    data = await request.json()
    GLOBAL_USDT = float(data.get("usdt", 0))
    # Sincronizamos el USDT en todos los pares
    for sym in SYMBOLS:
        real_portfolio[sym]["usdt"] = GLOBAL_USDT
    return {"status": "Saldo Actualizado", "usdt": GLOBAL_USDT}

@app.post("/registrar_trade")
async def registrar_trade(request: Request):
    global GLOBAL_USDT, real_portfolio
    data = await request.json()
    symbol = data.get("symbol")
    action = data.get("action")
    amount = float(data.get("amount", 0)) # Puede ser USDT o MONEDA
    price = float(data.get("price", 0))   # PRECIO MANUAL EXACTO
    
    if symbol not in real_portfolio: return {"error": "Moneda no válida"}
    if price <= 0: return {"error": "Precio inválido"}
    if amount <= 0: return {"error": "Cantidad inválida"}

    pf = real_portfolio[symbol]
    
    if action == "COMPRA":
        # Aquí 'amount' es USDT (Dólares que gastaste)
        usdt_spend = amount
        if GLOBAL_USDT < usdt_spend: return {"error": "Saldo USDT insuficiente"}
        
        # El sistema calcula cuántas monedas te dieron
        crypto_received = usdt_spend / price
        
        total_coins = pf["coin"] + crypto_received
        total_cost = (pf["coin"] * pf["avg_price"]) + usdt_spend
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else price
        pf["coin"] += crypto_received
        GLOBAL_USDT -= usdt_spend
        
    elif action == "VENTA":
        # Aquí 'amount' es CANTIDAD DE CRIPTO (Monedas que vendiste)
        crypto_spend = amount
        if pf["coin"] < crypto_spend * 0.9999: return {"error": "No tienes suficientes monedas"}
        
        # El sistema calcula cuántos dólares recibiste
        usdt_received = crypto_spend * price
        
        pf["coin"] -= crypto_spend
        if pf["coin"] < 0: pf["coin"] = 0
        GLOBAL_USDT += usdt_received
        if pf["coin"] <= 0.000001: pf["avg_price"] = 0.0 
        
    # Sincronizar saldo USDT global
    for s in SYMBOLS: real_portfolio[s]["usdt"] = GLOBAL_USDT
    
    return {"status": "OK", "nuevo_saldo": GLOBAL_USDT, "portfolio": real_portfolio}

# --- LÓGICA DE ALINEACIÓN DE DATOS ---
def obtener_historial_ajustado(symbol, precio_real_usuario):
    yahoo_symbol = symbol.replace("USDT", "-USD")
    ticker = yf.Ticker(yahoo_symbol)
    try:
        df = ticker.history(period="1d", interval="1m", auto_adjust=True)
        if df.empty:
            df = ticker.history(period="1d", interval="5m", auto_adjust=True)
        if not df.empty and precio_real_usuario > 0:
            ultimo_cierre_yahoo = df['Close'].iloc[-1]
            diferencia = precio_real_usuario - ultimo_cierre_yahoo
            df['Close'] += diferencia
            df['Open'] += diferencia
            df['High'] += diferencia
            df['Low'] += diferencia
            df = df.reset_index()
            df = df.rename(columns={"Close": "c", "High": "h", "Low": "l", "Open": "o", "Volume": "v"})
            return df
    except Exception as e: pass
    return pd.DataFrame()

def calcular_indicadores(df):
    if len(df) < 5: return df 
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

@app.get("/analisis")
async def get_analisis(symbol: str = "BTCUSDT", current_price: float = 0.0):
    global market_data_cache, real_portfolio
    if symbol not in SYMBOLS: symbol = "BTCUSDT"
    
    if current_price <= 0:
        cached = market_data_cache.get(symbol)
        if cached: current_price = cached["precio"]
    
    df = await run_in_threadpool(obtener_historial_ajustado, symbol, current_price)
    
    signal = "NEUTRAL"
    reasons = []
    
    if not df.empty:
        df = calcular_indicadores(df)
        current = df.iloc[-1]
        pf = real_portfolio[symbol]
        
        # Estrategia
        if pf["coin"] > 0 and pf["avg_price"] > 0:
            pnl = (current_price - pf["avg_price"]) / pf["avg_price"]
            if pnl <= -STOP_LOSS_PCT:
                signal = "VENTA FUERTE"
                reasons.append(f"🛑 STOP LOSS ({pnl*100:.2f}%)")
            elif pnl >= TAKE_PROFIT_PCT:
                signal = "VENTA FUERTE"
                reasons.append(f"💰 TAKE PROFIT ({pnl*100:.2f}%)")
                
        if signal == "NEUTRAL":
            try:
                if pf["coin"] == 0:
                    if current_price <= current['lower'] and current['rsi'] < 35:
                        signal = "COMPRA"
                        reasons.append("Soporte + RSI Bajo")
                elif pf["coin"] > 0:
                    if (current_price >= current['upper'] or current['rsi'] > 70):
                        signal = "VENTA"
                        reasons.append("Techo + RSI Alto")
            except: pass
    else:
        reasons.append("Esperando historial...")

    res = {
        "symbol": symbol, 
        "precio": current_price, 
        "decision": signal, 
        "detalles": reasons,
        "portfolio": real_portfolio, 
        "update_time": datetime.now().strftime("%H:%M:%S")
    }
    market_data_cache[symbol] = res
    return res

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return f.read()
    return "<h1>Error: No index.html</h1>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)