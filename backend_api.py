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
import numpy as np 

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CONFIGURACIÓN SWING INTRADÍA (5 MINUTOS) ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
GLOBAL_USDT = 0.0 

# ESTRATEGIA (Target 2.5% / Stop 1.5%)
STOP_LOSS_PCT = 0.015     
TAKE_PROFIT_PCT = 0.025   

real_portfolio = {
    sym: {"usdt": 0.0, "coin": 0.0, "avg_price": 0.0} 
    for sym in SYMBOLS
}
TRADE_LOG = [] 
market_data_cache = {} 

@app.post("/set_balance")
async def set_balance(request: Request):
    global GLOBAL_USDT, real_portfolio, TRADE_LOG
    data = await request.json()
    GLOBAL_USDT = float(data.get("usdt", 0))
    TRADE_LOG = [] 
    for sym in SYMBOLS:
        real_portfolio[sym]["usdt"] = GLOBAL_USDT
        real_portfolio[sym]["coin"] = 0.0
        real_portfolio[sym]["avg_price"] = 0.0
    return {"status": "Saldo Actualizado", "usdt": GLOBAL_USDT}

@app.post("/registrar_trade")
async def registrar_trade(request: Request):
    global GLOBAL_USDT, real_portfolio, TRADE_LOG
    data = await request.json()
    symbol = data.get("symbol")
    action = data.get("action")
    amount = float(data.get("amount", 0)) 
    price = float(data.get("price", 0))   
    
    if symbol not in real_portfolio: return {"error": "Moneda no válida"}
    if price <= 0: return {"error": "Precio inválido"}
    if amount <= 0: return {"error": "Cantidad inválida"}

    pf = real_portfolio[symbol]
    log_coin = 0.0
    log_usdt = 0.0
    
    if action == "COMPRA":
        if GLOBAL_USDT < amount: return {"error": "Saldo USDT insuficiente"}
        crypto_received = amount / price
        total_coins = pf["coin"] + crypto_received
        total_cost = (pf["coin"] * pf["avg_price"]) + amount
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else price
        pf["coin"] += crypto_received
        GLOBAL_USDT -= amount
        log_coin = crypto_received
        log_usdt = amount
        
    elif action == "VENTA":
        if pf["coin"] < amount * 0.9999: return {"error": "No tienes suficientes monedas"}
        usdt_received = amount * price
        pf["coin"] -= amount
        if pf["coin"] < 0: pf["coin"] = 0
        GLOBAL_USDT += usdt_received
        if pf["coin"] <= 0.000001: pf["avg_price"] = 0.0 
        log_coin = amount
        log_usdt = usdt_received
        
    for s in SYMBOLS: real_portfolio[s]["usdt"] = GLOBAL_USDT
    
    TRADE_LOG.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol, "action": action, "price": price,
        "coin": log_coin, "usdt": log_usdt
    })
    
    return {"status": "OK", "nuevo_saldo": GLOBAL_USDT, "portfolio": real_portfolio}

@app.get("/history/{symbol}")
def get_history(symbol: str):
    return [t for t in TRADE_LOG if t["symbol"] == symbol]

# --- LÓGICA DE ANÁLISIS ---
def obtener_historial_ajustado(symbol, precio_real_usuario):
    yahoo_symbol = symbol.replace("USDT", "-USD")
    ticker = yf.Ticker(yahoo_symbol)
    try:
        # Velas de 5m para filtrar ruido
        df = ticker.history(period="1d", interval="5m", auto_adjust=True)
        if df.empty:
            df = ticker.history(period="1d", interval="15m", auto_adjust=True)
        
        if not df.empty and precio_real_usuario > 0:
            ultimo_cierre = df['Close'].iloc[-1]
            diff = precio_real_usuario - ultimo_cierre
            df['Close'] += diff
            df['Open'] += diff
            df['High'] += diff
            df['Low'] += diff
            df = df.reset_index()
            df = df.rename(columns={"Close": "c", "High": "h", "Low": "l", "Open": "o", "Volume": "v"})
            return df
    except: pass
    return pd.DataFrame()

def calcular_indicadores(df):
    if len(df) < 20: return df 
    df['sma20'] = df['c'].rolling(window=20).mean()
    df['std20'] = df['c'].rolling(window=20).std()
    df['upper'] = df['sma20'] + (df['std20'] * 2)
    df['lower'] = df['sma20'] - (df['std20'] * 2)
    
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df = df.fillna(0)
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
    
    if not df.empty and len(df) > 20 and current_price > 0:
        df = calcular_indicadores(df)
        current = df.iloc[-1]
        pf = real_portfolio[symbol]
        
        rsi_val = current['rsi']
        
        if rsi_val <= 1 or np.isnan(rsi_val):
            signal = "NEUTRAL"
            reasons.append("Calculando indicadores...")
        else:
            # A) GESTIÓN DE POSICIÓN
            if pf["coin"] > 0 and pf["avg_price"] > 0:
                pnl_pct = (current_price - pf["avg_price"]) / pf["avg_price"]
                
                if pnl_pct <= -STOP_LOSS_PCT:
                    signal = "VENTA FUERTE"
                    reasons.append(f"🛑 STOP LOSS ({pnl_pct*100:.2f}%)")
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    signal = "VENTA FUERTE"
                    reasons.append(f"💰 TAKE PROFIT ({pnl_pct*100:.2f}%)")
                else:
                    if rsi_val > 75: 
                        signal = "VENTA"
                        reasons.append(f"RSI Extremo ({rsi_val:.0f})")
                    else:
                        signal = "MANTENER"
                        reasons.append(f"PnL: {pnl_pct*100:.2f}%")

            # B) BÚSQUEDA DE ENTRADA
            elif pf["coin"] == 0:
                # Estrategia Swing: Banda Inferior + RSI < 40
                if current_price <= current['lower'] and rsi_val < 40:
                    signal = "COMPRA"
                    reasons.append("Suelo Técnico (Bandas+RSI)")
                # Estrategia Sobreventa: RSI < 30
                elif rsi_val < 30:
                    signal = "COMPRA"
                    reasons.append(f"Sobreventa (RSI {rsi_val:.0f})")
                else:
                    signal = "NEUTRAL"
                    reasons.append(f"RSI: {rsi_val:.1f}")
    else:
        reasons.append("Sincronizando...")

    res = {
        "symbol": symbol, "precio": current_price, "decision": signal, 
        "detalles": reasons, "portfolio": real_portfolio,
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