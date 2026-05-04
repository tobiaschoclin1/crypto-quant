# 🚀 Crypto Quant Pro

Sistema de trading cuantitativo para criptomonedas con análisis técnico en tiempo real.

## 📊 Características

- Análisis técnico automatizado (EMA50, MACD, RSI)
- Gestión de portfolio multi-moneda (BTC, ETH, SOL, BNB, ADA)
- Sistema de señales inteligente (Compra/Venta/Mantener)
- Interfaz web responsive
- App móvil nativa (Flet)

## 🚀 Deployment

**Plataforma:** Render  
**URL:** https://crypto-quant-pro.onrender.com

### Keep-Alive (Evitar que se duerma)

**UptimeRobot:** Monitor cada 5 min → `/analisis?symbol=BTCUSDT`  
**Cron-job.org:** Monitor offset 2 min → `/analisis?symbol=ETHUSDT`

## 💻 Uso Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
python backend_api.py
```

Servidor en: `http://localhost:8000`

### App Móvil

```bash
# Configurar IP local en app_movil.py
API_URL = "http://TU_IP_LOCAL:8000/analisis"

# Ejecutar
python app_movil.py
```

## 📡 Endpoints API

- `GET /` - Interfaz web
- `GET /health` - Health check
- `GET /analisis?symbol=BTCUSDT` - Análisis técnico y señales
- `POST /set_balance` - Configurar balance inicial
- `POST /registrar_trade` - Registrar operación
- `GET /history/{symbol}` - Historial de trades

## 🛠️ Tecnologías

- FastAPI - Backend API
- Python 3.11 - Lenguaje principal
- Pandas/NumPy - Análisis de datos
- yFinance - Datos de mercado en tiempo real
- Flet - App móvil multiplataforma

## ⚠️ Advertencia

Este software es para propósitos educativos. No garantiza ganancias en operaciones reales. Los mercados cripto son altamente volátiles.

---

**Made with ❤️ for quantitative traders**
