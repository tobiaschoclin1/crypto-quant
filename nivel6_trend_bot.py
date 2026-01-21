import requests
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# --- CLASE BILLETERA (Idéntica al anterior) ---
class BilleteraVirtual:
    def __init__(self, saldo_inicial_usdt=1000):
        self.usdt = saldo_inicial_usdt
        self.btc = 0
        self.en_posicion = False
        self.precio_entrada = 0

    def comprar(self, precio, tiempo):
        if not self.en_posicion:
            self.btc = self.usdt / precio
            self.usdt = 0
            self.en_posicion = True
            self.precio_entrada = precio
            print(f"[VELA {tiempo}] 💎 COMPRA a ${precio:.2f}")

    def vender(self, precio, tiempo, motivo):
        if self.en_posicion:
            saldo_obtenido = self.btc * precio
            ganancia = saldo_obtenido - (self.btc * self.precio_entrada)
            pct = ((precio - self.precio_entrada) / self.precio_entrada) * 100
            self.usdt = saldo_obtenido
            self.btc = 0
            self.en_posicion = False
            icono = "✅" if ganancia > 0 else "❌"
            print(f"[VELA {tiempo}] 💵 VENTA ({motivo}) a ${precio:.2f} | PnL: {icono} {pct:.2f}% (${ganancia:.2f})")

    def saldo_total(self, precio_actual):
        return (self.btc * precio_actual) if self.en_posicion else self.usdt

# --- CEREBRO CON FILTRO SMA ---
def obtener_datos_con_indicadores(symbol="BTCUSDT", interval="15m", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(url, params=params).json()
    
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "v", "x", "y", "z", "w", "a", "b"])
    precios = df["close"].astype(float).values
    
    # Calculamos la Media Móvil Simple (SMA) de 50 periodos
    # Esto es el "Promedio" del precio en las últimas 50 velas
    sma_50 = df["close"].rolling(window=50).mean().values
    
    return precios, sma_50

def calcular_fisica(precios_window):
    # Suavizado y derivadas (igual que antes)
    suave = gaussian_filter1d(precios_window, sigma=2)
    vel = np.gradient(suave)
    acel = np.gradient(vel)
    return suave, vel, acel

# --- EJECUCIÓN ---
print("--- INICIANDO BOT DE TENDENCIA (15m) ---")

# 1. Bajamos datos de 15 minutos (Menos ruido, tendencias más largas)
precios_cierre, sma_50 = obtener_datos_con_indicadores(interval="15m", limit=500)
mi_billetera = BilleteraVirtual(1000)

# Empezamos desde la vela 55 para tener datos de la SMA
for t in range(55, len(precios_cierre)):
    
    # Datos actuales
    precio_real = precios_cierre[t]
    sma_actual = sma_50[t] # El valor promedio de las últimas 50 velas
    
    # Análisis de derivadas (ventana corta para reactividad)
    ventana = precios_cierre[t-20:t+1] 
    p, v, a = calcular_fisica(ventana)
    
    vel_actual = v[-1]
    vel_anterior = v[-2]
    acel_actual = a[-1]

    # --- ESTRATEGIA MEJORADA ---
    
    # 1. COMPRA: 
    #   - La velocidad cruza a positivo (el precio sube)
    #   - Y ADEMÁS: El precio está POR ENCIMA del promedio (Tendencia Alcista Sana)
    if (vel_anterior < 0 and vel_actual > 0) and (precio_real > sma_actual):
        mi_billetera.comprar(precio_real, t)

    # 2. VENTA (Stop Loss Dinámico):
    #   - Si el precio rompe hacia abajo el promedio, la fiesta se acabó. Vender.
    elif (precio_real < sma_actual) and mi_billetera.en_posicion:
        mi_billetera.vender(precio_real, t, "Cambio de Tendencia (SMA)")

    # 3. VENTA (Divergencia):
    #   - Si sube pero pierde fuerza (igual que antes)
    elif mi_billetera.en_posicion and (vel_actual < vel_anterior) and (acel_actual < -0.5):
        mi_billetera.vender(precio_real, t, "Divergencia/Techo")


# Resultados
saldo_final = mi_billetera.saldo_total(precios_cierre[-1])
rendimiento = ((saldo_final - 1000) / 1000) * 100

print("\n" + "="*30)
print(f"RESULTADO FINAL (Intervalo 15m)")
print(f"Saldo Inicial: $1000.00")
print(f"Saldo Final:   ${saldo_final:.2f}")
print(f"Rendimiento:   {rendimiento:.2f}%")
print("="*30)