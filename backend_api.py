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
import requests
from typing import Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CONFIGURACIÓN DE ESTRATEGIA (SIN PROMESAS DE RENTABILIDAD) ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
GLOBAL_USDT = 0.0 

# Riesgo base optimizado (se ajusta dinámicamente con ATR)
STOP_LOSS_PCT = 0.020      # 2% stop base (será multiplicado por ATR)
TAKE_PROFIT_PCT = 0.055    # 5.5% take profit base (risk-reward > 2.5:1)
TRAILING_STOP_PCT = 0.018  # 1.8% trailing para proteger ganancias
MIN_TRADE_USDT = 10.0      # Mínimo por operación

real_portfolio = {
    sym: {"usdt": 0.0, "coin": 0.0, "avg_price": 0.0, "highest_price": 0.0}
    for sym in SYMBOLS
}
TRADE_LOG = []
market_data_cache = {}
price_cache = {"data": {}, "last_update": None} 

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
    df['sma50'] = df['c'].rolling(window=50).mean()

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

    # Bollinger Bands
    df['bb_middle'] = df['c'].rolling(window=20).mean()
    bb_std = df['c'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volumen
    df['vol_sma20'] = df['v'].rolling(window=20).mean()
    df['vol_ratio'] = np.where(df['vol_sma20'] > 0, df['v'] / df['vol_sma20'], 1.0)
    df['vol_ema'] = df['v'].ewm(span=20, adjust=False).mean()

    # ADX (Average Directional Index) para medir fuerza de tendencia
    high_diff = df['h'].diff()
    low_diff = -df['l'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    df['plus_di'] = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False).mean() / df['atr14'])
    df['minus_di'] = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False).mean() / df['atr14'])
    df['adx'] = 100 * ((df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])).ewm(span=14, adjust=False).mean()

    # Support/Resistance (pivotes locales)
    df['resistance'] = df['h'].rolling(window=20).max()
    df['support'] = df['l'].rolling(window=20).min()
    df['dist_to_resistance'] = (df['resistance'] - df['c']) / df['c']
    df['dist_to_support'] = (df['c'] - df['support']) / df['c']

    # Stochastic RSI para timing más preciso
    rsi_min = df['rsi'].rolling(window=14).min()
    rsi_max = df['rsi'].rolling(window=14).max()
    df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min) * 100

    df = df.fillna(0)
    return df

def calcular_score_entrada(row: pd.Series, prev_row: pd.Series = None) -> int:
    score = 0

    # 1. Tendencia multi-timeframe (30 pts)
    if row['c'] > row['ema50']:
        score += 15
    if row['ema50'] > row['ema200']:
        score += 10
    if row['ema20'] > row['ema50']:
        score += 5

    # 2. Fuerza de tendencia (ADX) (15 pts)
    adx = row.get('adx', 0)
    if adx > 25:
        score += 15
    elif adx > 20:
        score += 8

    # 3. Momentum (20 pts)
    if row['macd'] > row['signal']:
        score += 12
    if prev_row is not None and row['macd_hist'] > prev_row['macd_hist']:
        score += 8

    # 4. RSI + Stochastic RSI (15 pts)
    if 40 <= row['rsi'] <= 60:
        score += 10
    elif 35 <= row['rsi'] <= 65:
        score += 5
    if row.get('stoch_rsi', 50) < 70:
        score += 5

    # 5. Posición en Bollinger Bands (10 pts)
    bb_pos = row.get('bb_position', 0.5)
    if 0.2 <= bb_pos <= 0.6:
        score += 10
    elif bb_pos < 0.3:
        score += 5

    # 6. Volumen (10 pts)
    if row['vol_ratio'] >= 1.2:
        score += 10
    elif row['vol_ratio'] >= 1.0:
        score += 5

    # 7. Distancia a soporte/resistencia (5 pts)
    dist_support = row.get('dist_to_support', 0)
    dist_resistance = row.get('dist_to_resistance', 0)
    if dist_support > 0.01 and dist_resistance > 0.02:
        score += 5

    # Penalizaciones
    if row['atr_pct'] > 0.07:
        score -= 15
    if row['rsi'] > 70:
        score -= 10
    if row.get('bb_width', 0.05) < 0.02:
        score -= 5

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
    adx = current.get('adx', 0)
    bb_position = current.get('bb_position', 0.5)

    score = calcular_score_entrada(current, prev)

    if rsi <= 1 or np.isnan(rsi):
        return "NEUTRAL", ["Calculando mercado..."], 0

    # Detectar régimen de mercado
    is_trending = adx > 25
    is_ranging = adx < 20
    bb_width = current.get('bb_width', 0.05)
    high_volatility = bb_width > 0.06 or atr_pct > 0.06

    # --- Gestión de posición ---
    if pf["coin"] > 0 and pf["avg_price"] > 0:
        pf["highest_price"] = max(pf.get("highest_price", 0.0), current_price)
        entry = pf["avg_price"]
        pnl_pct = (current_price - entry) / entry

        # Stop y TP dinámicos basados en ATR y mercado
        base_stop = max(STOP_LOSS_PCT, float(atr_pct) * 2.0)
        stop_dyn = base_stop * 1.2 if high_volatility else base_stop

        # Risk-reward ratio mejorado
        tp_dyn = stop_dyn * 2.5 if is_trending else stop_dyn * 2.0

        trailing_ref = pf.get("highest_price", current_price)
        trailing_pct = TRAILING_STOP_PCT * 1.5 if high_volatility else TRAILING_STOP_PCT
        trailing_trigger = trailing_ref * (1.0 - trailing_pct)

        # Condiciones de salida mejoradas
        trend_reversal = (current_price < ema50 and macd < sig_line and rsi < 45)
        momentum_weak = (current['macd_hist'] < prev['macd_hist'] and rsi < 50)
        overextended = rsi > 75 or bb_position > 0.95

        if pnl_pct <= -stop_dyn:
            signal = "VENTA FUERTE"
            reasons.append(f"🛑 Stop loss ({pnl_pct*100:.2f}%)")
        elif pnl_pct >= tp_dyn:
            signal = "VENTA FUERTE"
            reasons.append(f"💰 Take profit ({pnl_pct*100:.2f}%)")
        elif overextended and pnl_pct > 0.01:
            signal = "VENTA"
            reasons.append(f"⚠️ Mercado sobreextendido, asegurar ganancia ({pnl_pct*100:.2f}%)")
        elif current_price <= trailing_trigger and pnl_pct > 0.01:
            signal = "VENTA"
            reasons.append(f"📉 Trailing stop ({pnl_pct*100:.2f}%)")
        elif trend_reversal:
            signal = "VENTA"
            reasons.append(f"🔄 Reversión de tendencia confirmada ({pnl_pct*100:.2f}%)")
        elif momentum_weak and pnl_pct > 0.015:
            signal = "VENTA"
            reasons.append(f"⚡ Momentum debilitándose, asegurar ganancia ({pnl_pct*100:.2f}%)")
        else:
            signal = "MANTENER"
            reasons.append(f"📊 Posición activa: {pnl_pct*100:.2f}%")
            if is_trending:
                reasons.append("✅ Tendencia fuerte continúa")
            if current_price > ema20:
                reasons.append("✅ Por encima de EMA20")

        return signal, reasons, score

    # --- Búsqueda de entrada ---
    if usdt_global < MIN_TRADE_USDT:
        return "NEUTRAL", ["Sin saldo suficiente para operar"], score

    # Condiciones de entrada más estrictas
    trend_strong = (current_price > ema50) and (ema50 > ema200) and (ema20 > ema50)
    trend_ok = (current_price > ema50) and (ema50 > ema200)

    pullback_quality = (
        (ema20 <= current_price <= ema50 * 1.02) and
        (40 <= rsi <= 55) and
        (0.2 <= bb_position <= 0.5)
    )

    breakout_setup = (
        (current_price > ema20) and
        (bb_position > 0.6) and
        (vol_ratio > 1.2)
    )

    momentum_strong = (macd > sig_line) and (current['macd_hist'] > prev['macd_hist'])
    momentum_ok = (macd > sig_line)

    volume_strong = vol_ratio >= 1.15
    volume_ok = vol_ratio >= 0.95

    not_overbought = rsi < 68 and bb_position < 0.85

    # Señales de entrada jerárquicas
    if score >= 80 and trend_strong and momentum_strong and volume_strong and is_trending and not_overbought:
        signal = "COMPRA FUERTE"
        reasons.append(f"🚀 Setup excepcional (score {score}/100)")
        reasons.append(f"ADX: {adx:.1f} (tendencia fuerte)")
    elif score >= 70 and trend_ok and pullback_quality and momentum_strong and volume_ok:
        signal = "COMPRA FUERTE"
        reasons.append(f"📈 Pullback de alta calidad (score {score}/100)")
        reasons.append("Zona de compra óptima")
    elif score >= 65 and trend_strong and breakout_setup and momentum_ok:
        signal = "COMPRA"
        reasons.append(f"⚡ Breakout con volumen (score {score}/100)")
    elif score >= 60 and trend_ok and momentum_ok and volume_ok and not_overbought and not is_ranging:
        signal = "COMPRA"
        reasons.append(f"✅ Setup favorable (score {score}/100)")
    else:
        signal = "NEUTRAL"
        reasons.append(f"⏸ Esperando mejor oportunidad (score {score}/100)")

        if not trend_ok:
            reasons.append("❌ Tendencia no confirmada")
        elif is_ranging:
            reasons.append("⚠️ Mercado lateral, evitar entradas")
        elif not not_overbought:
            reasons.append("⚠️ Mercado sobrecomprado")
        elif not momentum_ok:
            reasons.append("⏳ Esperando confirmación de momentum")
        elif not volume_ok:
            reasons.append("📉 Volumen insuficiente")
        else:
            reasons.append("⏳ Condiciones no óptimas")

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
        return {"trades": 0, "win_rate": 0.0, "net_return_pct": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0}

    in_pos = False
    entry = 0.0
    highest = 0.0
    pnls = []

    for i in range(210, len(df)):
        row = df.iloc[i]
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
            if sell_by_signal:
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
        return {"trades": 0, "win_rate": 0.0, "net_return_pct": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0}

    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p <= 0]
    win_rate = (len(wins) / trades) * 100.0
    net = float(np.sum(pnls)) * 100.0
    gross_win = float(np.sum(wins))
    gross_loss = float(np.sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 999.0
    avg_win = (gross_win / len(wins)) * 100.0 if wins else 0.0
    avg_loss = (gross_loss / len(losses)) * 100.0 if losses else 0.0

    return {
        "trades": trades,
        "win_rate": round(win_rate, 2),
        "net_return_pct": round(net, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2)
    }

@app.get("/prices")
def get_current_prices():
    """Obtiene precios con múltiples fuentes y caché corto"""
    global price_cache
    from datetime import datetime
    import traceback

    # Caché de 6 MINUTOS (para que nunca expire entre pings de UptimeRobot que son cada 5 min)
    CACHE_DURATION = 360  # 6 minutos en segundos
    now = datetime.now()
    if price_cache["last_update"]:
        time_diff = (now - price_cache["last_update"]).total_seconds()
        if time_diff < CACHE_DURATION and price_cache["data"] and any(p > 0 for p in price_cache["data"].values()):
            print(f"→ Cache hit (age: {time_diff:.1f}s / {CACHE_DURATION}s)")
            return price_cache["data"]

    prices = {}
    print(f"\n=== Fetching fresh prices at {now.strftime('%H:%M:%S')} ===")

    # FUENTE 1: CoinGecko Simple API (más generosa con rate limits)
    try:
        print("→ Trying CoinGecko...")
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana,binancecoin,cardano&vs_currencies=usd"
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        print(f"  CoinGecko status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  CoinGecko response: {data}")

            mapping = {
                "bitcoin": "BTCUSDT",
                "ethereum": "ETHUSDT",
                "solana": "SOLUSDT",
                "binancecoin": "BNBUSDT",
                "cardano": "ADAUSDT"
            }

            for coin_id, symbol in mapping.items():
                if coin_id in data and 'usd' in data[coin_id]:
                    prices[symbol] = float(data[coin_id]['usd'])

            if len(prices) >= 3:
                print(f"✓ CoinGecko SUCCESS: {len(prices)}/5 prices")
                for sym, price in prices.items():
                    print(f"  {sym}: ${price:.2f}")
                price_cache["data"] = prices
                price_cache["last_update"] = now
                return prices
        else:
            print(f"  CoinGecko failed: {response.text[:200]}")
    except Exception as e:
        print(f"✗ CoinGecko error: {str(e)}")
        traceback.print_exc()

    # FUENTE 2: CoinCap.io
    prices = {}  # Reset
    try:
        print("→ Trying CoinCap.io...")
        coin_ids = ["bitcoin", "ethereum", "solana", "binance-coin", "cardano"]
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]

        for coin_id, symbol in zip(coin_ids, symbols):
            try:
                url = f"https://api.coincap.io/v2/assets/{coin_id}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, timeout=10, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'priceUsd' in data['data']:
                        price = float(data['data']['priceUsd'])
                        prices[symbol] = price
                        print(f"  {symbol}: ${price:.2f}")
            except Exception as e:
                print(f"  {symbol} failed: {e}")
                continue

        if len(prices) >= 3:
            print(f"✓ CoinCap SUCCESS: {len(prices)}/5 prices")
            price_cache["data"] = prices
            price_cache["last_update"] = now
            return prices
    except Exception as e:
        print(f"✗ CoinCap error: {str(e)}")

    # FUENTE 3: Binance API (puede estar bloqueada en algunas regiones)
    prices = {}  # Reset
    try:
        print("→ Trying Binance...")
        symbols_binance = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        url = "https://api.binance.com/api/v3/ticker/price"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        response = requests.get(url, timeout=15, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                if item['symbol'] in symbols_binance:
                    prices[item['symbol']] = float(item['price'])

            if len(prices) >= 3:
                print(f"✓ Binance SUCCESS: {len(prices)}/5 prices")
                price_cache["data"] = prices
                price_cache["last_update"] = now
                return prices
        else:
            print(f"  Binance status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"✗ Binance error: {str(e)}")

    # Fallback: usar caché aunque sea viejo (mejor que 0s)
    if price_cache["data"] and any(p > 0 for p in price_cache["data"].values()):
        age = (now - price_cache["last_update"]).total_seconds()
        print(f"⚠ Using stale cache (age: {age:.0f}s)")
        return price_cache["data"]

    # Último recurso: retornar 0s
    print("✗ ALL SOURCES FAILED - returning zeros")
    return {sym: 0 for sym in SYMBOLS}

price_debug_log = []

@app.get("/health")
async def health_check():
    """Endpoint de health check para verificar que la API funciona

    También precarga precios en background para mantener el caché caliente.
    Esto asegura que cuando un usuario visite la app, los precios ya estén disponibles.
    """
    import asyncio
    # Ejecutar get_current_prices() en background para calentar el caché
    # No bloqueamos la respuesta para que UptimeRobot reciba OK inmediatamente
    asyncio.create_task(asyncio.to_thread(get_current_prices))

    return {
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": SYMBOLS
    }

@app.get("/test_apis")
async def test_apis_simple():
    """Prueba directa de las APIs sin la lógica compleja"""
    import traceback
    results = {}

    # Test 1: CoinGecko
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
        response = requests.get(url, timeout=10, headers={'Accept': 'application/json'})
        results["coingecko"] = {
            "status": response.status_code,
            "data": response.json() if response.status_code == 200 else response.text[:300]
        }
    except Exception as e:
        results["coingecko"] = {"error": str(e), "traceback": traceback.format_exc()}

    # Test 2: Binance
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=10)
        results["binance"] = {
            "status": response.status_code,
            "data": response.json() if response.status_code == 200 else response.text[:300]
        }
    except Exception as e:
        results["binance"] = {"error": str(e), "traceback": traceback.format_exc()}

    # Test 3: CoinCap
    try:
        url = "https://api.coincap.io/v2/assets/bitcoin"
        response = requests.get(url, timeout=10)
        results["coincap"] = {
            "status": response.status_code,
            "data": response.json() if response.status_code == 200 else response.text[:300]
        }
    except Exception as e:
        results["coincap"] = {"error": str(e), "traceback": traceback.format_exc()}

    return results

@app.get("/force_refresh")
async def force_refresh_prices():
    """Fuerza la actualización de precios y retorna el resultado"""
    global price_cache
    from fastapi.concurrency import run_in_threadpool

    # Limpiar caché para forzar refresh
    price_cache["data"] = {}
    price_cache["last_update"] = None

    # Obtener precios frescos
    prices = await run_in_threadpool(get_current_prices)

    return {
        "success": any(p > 0 for p in prices.values()) if prices else False,
        "prices": prices,
        "cache_updated": price_cache["last_update"].isoformat() if price_cache["last_update"] else None
    }

@app.get("/debug_prices")
def debug_prices():
    """Endpoint de debug que intenta obtener precios y retorna info detallada"""
    import traceback
    result = {"attempts": []}

    # Intento 1: CryptoCompare
    try:
        url = "https://min-api.cryptocompare.com/data/pricemulti?fsyms=BTC,ETH&tsyms=USD"
        response = requests.get(url, timeout=10)
        result["attempts"].append({
            "source": "CryptoCompare",
            "url": url,
            "status": response.status_code,
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else response.text[:200]
        })
    except Exception as e:
        result["attempts"].append({
            "source": "CryptoCompare",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

    # Intento 2: CoinGecko
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        result["attempts"].append({
            "source": "CoinGecko",
            "url": url,
            "status": response.status_code,
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else response.text[:200]
        })
    except Exception as e:
        result["attempts"].append({
            "source": "CoinGecko",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

    # Intento 3: Prueba de conectividad básica
    try:
        url = "https://httpbin.org/get"
        response = requests.get(url, timeout=10)
        result["attempts"].append({
            "source": "httpbin.org (connectivity test)",
            "status": response.status_code,
            "success": response.status_code == 200
        })
    except Exception as e:
        result["attempts"].append({
            "source": "httpbin.org",
            "error": str(e)
        })

    return result

@app.get("/test_prices")
async def test_prices_debug():
    """Endpoint de diagnóstico para ver por qué fallan los precios"""
    import traceback
    results = {}

    # Test 1: yfinance con BTC
    try:
        ticker = yf.Ticker("BTC-USD")
        results["yfinance_ticker_created"] = True

        try:
            fast_info = ticker.fast_info
            price = fast_info.get('lastPrice', None)
            results["yfinance_fast_info"] = {"price": price, "success": price is not None}
        except Exception as e:
            results["yfinance_fast_info"] = {"error": str(e), "traceback": traceback.format_exc()}

        try:
            hist = ticker.history(period="1d", interval="5m")
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                results["yfinance_history"] = {"price": price, "rows": len(hist)}
            else:
                results["yfinance_history"] = {"error": "Empty dataframe"}
        except Exception as e:
            results["yfinance_history"] = {"error": str(e), "traceback": traceback.format_exc()}

    except Exception as e:
        results["yfinance_ticker_created"] = False
        results["yfinance_error"] = str(e)
        results["yfinance_traceback"] = traceback.format_exc()

    # Test 2: Binance API
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                                  timeout=aiohttp.ClientTimeout(total=10)) as response:
                results["binance_status"] = response.status
                if response.status == 200:
                    data = await response.json()
                    results["binance_data"] = data
                else:
                    results["binance_error"] = await response.text()
    except Exception as e:
        results["binance_error"] = str(e)
        results["binance_traceback"] = traceback.format_exc()

    return results

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

@app.on_event("startup")
async def startup_warmup():
    """Pre-carga precios al iniciar para evitar mostrar 0s en cold start"""
    print("🔥 Warmup: Pre-cargando precios al arrancar...")
    try:
        # Ejecutar en threadpool porque get_current_prices() es síncrona y hace requests bloqueantes
        from fastapi.concurrency import run_in_threadpool
        prices = await run_in_threadpool(get_current_prices)
        if prices and any(p > 0 for p in prices.values()):
            print(f"✅ Warmup exitoso: {sum(1 for p in prices.values() if p > 0)}/5 precios cargados")
            for sym, price in prices.items():
                if price > 0:
                    print(f"   {sym}: ${price:,.2f}")
        else:
            print("⚠️ Warmup: No se pudieron cargar precios al arrancar")
    except Exception as e:
        import traceback
        print(f"❌ Error en warmup: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)