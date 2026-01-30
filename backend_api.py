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

# --- CONFIGURACIÓN MOMENTUM (4-6 HORAS) ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
GLOBAL_USDT = 0.0 

# BUSCAMOS MOVIMIENTOS RÁPIDOS Y PROBABLES
# Ganancia objetivo: 1.3% (Realista en 2-3 velas de 15m)
# Stop Loss: 0.8% (Muy corto, si falla salimos ya)
STOP_LOSS_PCT = 0.008     
TAKE_PROFIT_PCT = 0.013   

real_portfolio = {
    sym: {"usdt": 0.0, "coin": 0.0, "avg_price": 0.0} 
    for sym in SYMBOLS
}
TRADE_LOG = [] 
market_data_cache = {} 

@app.post("/set_balance")
async def set_balance(request: Request):
    global GLOBAL_USDT, real_portfolio
    data = await request.json()
    GLOBAL_USDT = float(data.get("usdt", 0))
    for sym in SYMBOLS:
        real_portfolio[sym]["usdt"] = GLOBAL_USDT
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

# --- LÓGICA DE DATOS ---
def obtener_historial_ajustado(symbol, precio_real_usuario):
    yahoo_symbol = symbol.replace("USDT", "-USD")
    ticker = yf.Ticker(yahoo_symbol)
    try:
        # Volvemos a 5m para tener agilidad, pero filtraremos mejor
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
    if len(df) < 26: return df 
    
    # EMA 20 (Tendencia Corta - Gatillo rápido)
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
    # EMA 50 (Tendencia Media - Filtro de seguridad)
    df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
    
    # MACD Clásico
    ema12 = df['c'].ewm(span=12, adjust=False).mean()
    ema26 = df['c'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # RSI
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
    
    if not df.empty and len(df) > 26 and current_price > 0:
        df = calcular_indicadores(df)
        current = df.iloc[-1]
        pf = real_portfolio[symbol]
        
        rsi = current['rsi']
        ema20 = current['ema20']
        ema50 = current['ema50']
        macd = current['macd']
        sig_line = current['signal']
        
        if rsi <= 1 or np.isnan(rsi):
            signal = "NEUTRAL"
            reasons.append("Calculando...")
        else:
            # A) GESTIÓN DE VENTA
            if pf["coin"] > 0 and pf["avg_price"] > 0:
                pnl_pct = (current_price - pf["avg_price"]) / pf["avg_price"]
                
                # Stop Loss Corto (Cortar pérdidas rápido)
                if pnl_pct <= -STOP_LOSS_PCT:
                    signal = "VENTA FUERTE"
                    reasons.append(f"🛑 STOP LOSS ({pnl_pct*100:.2f}%)")
                
                # Take Profit Realista
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    signal = "VENTA FUERTE"
                    reasons.append(f"💰 TAKE PROFIT ({pnl_pct*100:.2f}%)")
                
                else:
                    # SALIDA INTELIGENTE (Trailing Stop Manual)
                    # Si ya ganamos algo (>0.5%) y el precio cae debajo de la EMA20, huimos.
                    if current_price < ema20 and pnl_pct > 0.005:
                        signal = "VENTA"
                        reasons.append("Perdida de tendencia (EMA20)")
                    elif rsi > 70: # Sobrecompra, mejor asegurar
                        signal = "VENTA"
                        reasons.append("RSI Alto (>70)")
                    else:
                        signal = "MANTENER"
                        reasons.append(f"PnL: {pnl_pct*100:.2f}%")

            # B) GESTIÓN DE COMPRA (MOMENTUM)
            elif pf["coin"] == 0:
                # REGLAS DE ORO PARA ENTRAR (Filtrar basura):
                # 1. TENDENCIA: Precio > EMA50 (Solo operamos a favor de la corriente)
                # 2. IMPULSO: Precio > EMA20 (Está subiendo ahora mismo)
                # 3. FUERZA: RSI > 50 (Hay compradores) PERO RSI < 70 (No llegamos tarde)
                # 4. MACD: Histograma positivo (MACD > Signal)
                
                trend_ok = current_price > ema50
                momentum_ok = current_price > ema20
                rsi_ok = 50 < rsi < 70
                macd_ok = macd > sig_line
                
                if trend_ok and momentum_ok and rsi_ok and macd_ok:
                    signal = "COMPRA"
                    reasons.append("🚀 IMPULSO CONFIRMADO")
                
                else:
                    signal = "NEUTRAL"
                    if not trend_ok: reasons.append("Esperando Tendencia (Bajo EMA50)")
                    elif not rsi_ok: reasons.append(f"RSI fuera de rango ({rsi:.0f})")
                    elif not momentum_ok: reasons.append("Precio débil (Bajo EMA20)")
                    else: reasons.append("Esperando cruce MACD")
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