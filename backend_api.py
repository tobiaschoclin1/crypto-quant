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
from typing import Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CONFIGURACIÓN DE ESTRATEGIA (SIN PROMESAS DE RENTABILIDAD) ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
GLOBAL_USDT = 0.0 

# Riesgo base (se ajusta con ATR en cada símbolo)
STOP_LOSS_PCT = 0.025
TAKE_PROFIT_PCT = 0.050
TRAILING_STOP_PCT = 0.02
MIN_TRADE_USDT = 10.0

real_portfolio = {
    sym: {"usdt": 0.0, "coin": 0.0, "avg_price": 0.0, "highest_price": 0.0}
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
        if amount < MIN_TRADE_USDT: return {"error": f"Compra mínima: {MIN_TRADE_USDT:.0f} USDT"}
        crypto_received = amount / price
        total_coins = pf["coin"] + crypto_received
        total_cost = (pf["coin"] * pf["avg_price"]) + amount
        pf["avg_price"] = total_cost / total_coins if total_coins > 0 else price
        pf["coin"] += crypto_received
        pf["highest_price"] = max(pf.get("highest_price", 0.0), price)
        GLOBAL_USDT -= amount
        log_coin = crypto_received
        log_usdt = amount
        
    elif action == "VENTA":
        if pf["coin"] < amount * 0.9999: return {"error": "No tienes suficientes monedas"}
        usdt_received = amount * price
        pf["coin"] -= amount
        if pf["coin"] < 0: pf["coin"] = 0
        GLOBAL_USDT += usdt_received
        if pf["coin"] <= 0.000001:
            pf["avg_price"] = 0.0
            pf["highest_price"] = 0.0
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
        # Más contexto para filtrar ruido y reducir señales falsas.
        df = ticker.history(period="30d", interval="1h", auto_adjust=True)
        if df.empty:
            df = ticker.history(period="90d", interval="1h", auto_adjust=True)
        
        if not df.empty:
            if precio_real_usuario > 0:
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
    if len(df) < 210:
        return df 
    
    # Tendencia
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['c'].ewm(span=200, adjust=False).mean()
    
    # Momentum
    ema12 = df['c'].ewm(span=12, adjust=False).mean()
    ema26 = df['c'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # RSI
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Volatilidad (ATR)
    prev_close = df['c'].shift(1)
    tr = pd.concat([
        (df['h'] - df['l']).abs(),
        (df['h'] - prev_close).abs(),
        (df['l'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr14'] / df['c']

    # Volumen relativo
    df['vol_sma20'] = df['v'].rolling(window=20).mean()
    df['vol_ratio'] = np.where(df['vol_sma20'] > 0, df['v'] / df['vol_sma20'], 1.0)
    
    df = df.fillna(0)
    return df

def calcular_score_entrada(row: pd.Series) -> int:
    score = 0

    # Tendencia (45)
    if row['c'] > row['ema50']:
        score += 25
    if row['ema50'] > row['ema200']:
        score += 20

    # Momentum (30)
    if row['macd'] > row['signal']:
        score += 20
    if row['macd_hist'] > 0:
        score += 10

    # RSI (15)
    if 45 <= row['rsi'] <= 62:
        score += 15
    elif 40 <= row['rsi'] <= 70:
        score += 8

    # Volumen (10)
    if row['vol_ratio'] >= 1.05:
        score += 10

    # Penalización por exceso de volatilidad
    if row['atr_pct'] > 0.06:
        score -= 10

    return int(max(0, min(100, score)))

def estrategia_decision(df: pd.DataFrame, current_price: float, pf: Dict[str, Any], usdt_global: float):
    signal = "NEUTRAL"
    reasons = []

    current = df.iloc[-1]
    prev = df.iloc[-2]

    ema20 = current['ema20']
    ema50 = current['ema50']
    ema200 = current['ema200']
    macd = current['macd']
    sig_line = current['signal']
    rsi = current['rsi']
    atr_pct = current['atr_pct']
    vol_ratio = current['vol_ratio']

    score = calcular_score_entrada(current)

    if rsi <= 1 or np.isnan(rsi):
        return "NEUTRAL", ["Calculando mercado..."], 0

    # --- Gestión de posición ---
    if pf["coin"] > 0 and pf["avg_price"] > 0:
        pf["highest_price"] = max(pf.get("highest_price", 0.0), current_price)
        entry = pf["avg_price"]
        pnl_pct = (current_price - entry) / entry

        # Stop dinámico por volatilidad actual
        stop_dyn = max(STOP_LOSS_PCT, float(atr_pct) * 1.8)
        tp_dyn = max(TAKE_PROFIT_PCT, stop_dyn * 1.8)

        trailing_ref = pf.get("highest_price", current_price)
        trailing_trigger = trailing_ref * (1.0 - TRAILING_STOP_PCT)

        if pnl_pct <= -stop_dyn:
            signal = "VENTA FUERTE"
            reasons.append(f"🛑 Stop dinámico activado ({pnl_pct*100:.2f}%)")
        elif pnl_pct >= tp_dyn:
            signal = "VENTA FUERTE"
            reasons.append(f"💰 Take profit dinámico ({pnl_pct*100:.2f}%)")
        elif current_price <= trailing_trigger and pnl_pct > 0.005:
            signal = "VENTA"
            reasons.append("📉 Trailing stop: proteger ganancia")
        elif (current_price < ema50 and macd < sig_line and rsi < 46):
            signal = "VENTA"
            reasons.append("Cambio de tendencia confirmado")
        else:
            signal = "MANTENER"
            reasons.append(f"Posición en gestión: {pnl_pct*100:.2f}%")
            if current_price > ema20:
                reasons.append("Tendencia de corto plazo favorable")

        return signal, reasons, score

    # --- Búsqueda de entrada ---
    if usdt_global < MIN_TRADE_USDT:
        return "NEUTRAL", ["Sin saldo suficiente para operar"], score

    trend_ok = (current_price > ema50) and (ema50 > ema200)
    pullback_ok = (current_price >= ema20) and (40 <= rsi <= 65)
    momentum_ok = (macd > sig_line) and (current['macd_hist'] >= prev['macd_hist'])
    volume_ok = vol_ratio >= 0.95

    if score >= 75 and trend_ok and momentum_ok and volume_ok:
        signal = "COMPRA FUERTE"
        reasons.append(f"Setup de alta probabilidad (score {score}/100)")
    elif score >= 60 and trend_ok and pullback_ok and momentum_ok:
        signal = "COMPRA"
        reasons.append(f"Setup favorable (score {score}/100)")
    else:
        signal = "NEUTRAL"
        reasons.append(f"Sin ventaja estadística (score {score}/100)")
        if not trend_ok:
            reasons.append("Esperando tendencia alcista clara")
        elif not momentum_ok:
            reasons.append("Esperando confirmación MACD")
        elif not volume_ok:
            reasons.append("Volumen débil")

    return signal, reasons, score

@app.get("/analisis")
async def get_analisis(symbol: str = "BTCUSDT", current_price: float = 0.0):
    global market_data_cache, real_portfolio, GLOBAL_USDT
    if symbol not in SYMBOLS: symbol = "BTCUSDT"
    
    if current_price <= 0:
        cached = market_data_cache.get(symbol)
        if cached: current_price = cached["precio"]

    df = await run_in_threadpool(obtener_historial_ajustado, symbol, current_price)
    
    signal = "NEUTRAL"
    reasons = []
    score = 0
    
    if not df.empty and len(df) > 210 and current_price > 0:
        df = calcular_indicadores(df)
        pf = real_portfolio[symbol]
        signal, reasons, score = estrategia_decision(df, current_price, pf, GLOBAL_USDT)
    else:
        reasons.append("Sincronizando...")

    res = {
        "symbol": symbol, "precio": current_price, "decision": signal, 
        "detalles": reasons, "portfolio": real_portfolio,
        "score": score,
        "update_time": datetime.now().strftime("%H:%M:%S")
    }
    market_data_cache[symbol] = res
    return res

def _run_backtest_symbol(df: pd.DataFrame):
    df = calcular_indicadores(df.copy())
    if len(df) < 220:
        return {"trades": 0, "win_rate": 0.0, "net_return_pct": 0.0, "profit_factor": 0.0}

    in_pos = False
    entry = 0.0
    highest = 0.0
    pnls = []

    for i in range(210, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(row['c'])

        fake_pf = {"coin": 1.0 if in_pos else 0.0, "avg_price": entry if in_pos else 0.0, "highest_price": highest}
        decision, _, score = estrategia_decision(df.iloc[:i+1], price, fake_pf, 1000.0)

        if not in_pos and (decision.startswith("COMPRA") and score >= 60):
            in_pos = True
            entry = price
            highest = price
            continue

        if in_pos:
            highest = max(highest, price)
            sell_by_signal = decision.startswith("VENTA")
            momentum_lost = row['macd_hist'] < prev['macd_hist'] and row['rsi'] > 70
            if sell_by_signal or momentum_lost:
                pnl = (price - entry) / entry
                pnls.append(pnl)
                in_pos = False
                entry = 0.0
                highest = 0.0

    if in_pos:
        pnl = (float(df.iloc[-1]['c']) - entry) / entry
        pnls.append(pnl)

    trades = len(pnls)
    if trades == 0:
        return {"trades": 0, "win_rate": 0.0, "net_return_pct": 0.0, "profit_factor": 0.0}

    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p <= 0]
    win_rate = (len(wins) / trades) * 100.0
    net = float(np.sum(pnls)) * 100.0
    gross_win = float(np.sum(wins))
    gross_loss = float(np.sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 999.0

    return {
        "trades": trades,
        "win_rate": round(win_rate, 2),
        "net_return_pct": round(net, 2),
        "profit_factor": round(profit_factor, 2)
    }

@app.get("/backtest")
async def backtest(symbol: str = "BTCUSDT"):
    if symbol not in SYMBOLS:
        symbol = "BTCUSDT"

    yahoo_symbol = symbol.replace("USDT", "-USD")
    ticker = yf.Ticker(yahoo_symbol)
    try:
        df = await run_in_threadpool(lambda: ticker.history(period="180d", interval="1h", auto_adjust=True))
        if df.empty:
            return {"error": "Sin datos para backtest"}
        df = df.reset_index().rename(columns={"Close": "c", "High": "h", "Low": "l", "Open": "o", "Volume": "v"})
        stats = _run_backtest_symbol(df)
        return {"symbol": symbol, "timeframe": "1h", **stats}
    except Exception as e:
        return {"error": f"Backtest falló: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
def read_root():
    # Detecta si corre en modo script normal o compilado
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    html_path = os.path.join(base_path, "index.html")
    
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f: return f.read()
    # Fallback para desarrollo local simple sin pyinstaller
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return f.read()
        
    return "<h1>Error: No index.html found</h1>"

# Import necesario para la deteccion de rutas
import sys

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)