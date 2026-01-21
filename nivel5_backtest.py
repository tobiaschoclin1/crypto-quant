import requests
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# --- 1. CLASE BILLETERA (Tu cuenta bancaria simulada) ---
class BilleteraVirtual:
    def __init__(self, saldo_inicial_usdt=1000):
        self.usdt = saldo_inicial_usdt
        self.btc = 0
        self.en_posicion = False  # ¿Tenemos BTC comprado?
        self.precio_entrada = 0

    def comprar(self, precio, tiempo):
        if not self.en_posicion:
            # Compramos todo lo que tenemos
            cantidad_btc = self.usdt / precio
            self.btc = cantidad_btc
            self.usdt = 0
            self.en_posicion = True
            self.precio_entrada = precio
            print(f"[MIN {tiempo}] 💎 COMPRA EJECUTADA a ${precio:.2f}")

    def vender(self, precio, tiempo, motivo):
        if self.en_posicion:
            # Vendemos todo
            saldo_obtenido = self.btc * precio
            ganancia = saldo_obtenido - (self.btc * self.precio_entrada)
            porcentaje = ((precio - self.precio_entrada) / self.precio_entrada) * 100
            
            self.usdt = saldo_obtenido
            self.btc = 0
            self.en_posicion = False
            
            icono = "✅" if ganancia > 0 else "❌"
            print(f"[MIN {tiempo}] 💵 VENTA ({motivo}) a ${precio:.2f} | PnL: {icono} {porcentaje:.2f}% (${ganancia:.2f})")

    def saldo_total(self, precio_actual):
        if self.en_posicion:
            return self.btc * precio_actual
        return self.usdt

# --- 2. LÓGICA DEL BOT (Igual al Nivel 4) ---
def analizar_mercado(precios_raw):
    if len(precios_raw) < 5: return precios_raw, np.zeros_like(precios_raw), np.zeros_like(precios_raw)
    precios = gaussian_filter1d(precios_raw, sigma=2)
    velocidad = np.gradient(precios)
    aceleracion = np.gradient(velocidad)
    return precios, velocidad, aceleracion

# --- 3. EJECUCIÓN ---
print("--- INICIANDO BACKTESTING ---")
print("Capital Inicial: $1000 USDT")

# Descargar datos (Pedimos más datos para tener más juego: 200 minutos)
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 200}
data = requests.get(url, params=params).json()
precios_cierre = np.array([float(x[4]) for x in data])

mi_billetera = BilleteraVirtual(1000)

# Simulamos el paso del tiempo
for t in range(20, len(precios_cierre)):
    # Datos hasta el momento 't'
    datos_ventana = precios_cierre[:t]
    p, v, a = analizar_mercado(datos_ventana)
    
    precio_actual = p[-1] # Precio suavizado para cálculos
    precio_real_ejecucion = precios_cierre[t] # Precio real al que compramos/vendemos
    
    vel_actual = v[-1]
    vel_anterior = v[-2]
    acel_actual = a[-1]
    
    # --- ESTRATEGIA ---
    
    # 1. COMPRA: Cruce de velocidad hacia arriba
    if vel_anterior < 0 and vel_actual > 0:
        mi_billetera.comprar(precio_real_ejecucion, t)
        
    # 2. VENTA: Divergencia (Techo)
    # Ajustamos un poco la sensibilidad (-0.3) para asegurar la salida
    elif (precio_actual > p[-3]) and (vel_actual < vel_anterior) and (acel_actual < -0.3):
        mi_billetera.vender(precio_real_ejecucion, t, "Divergencia")
        
    # 3. VENTA: Stop Loss (Caída fuerte)
    elif vel_actual < -15:
        mi_billetera.vender(precio_real_ejecucion, t, "Pánico")

# Resumen Final
saldo_final = mi_billetera.saldo_total(precios_cierre[-1])
rendimiento = ((saldo_final - 1000) / 1000) * 100

print("\n" + "="*30)
print(f"RESULTADO FINAL")
print(f"Saldo Inicial: $1000.00")
print(f"Saldo Final:   ${saldo_final:.2f}")
print(f"Rendimiento:   {rendimiento:.2f}%")
print("="*30)