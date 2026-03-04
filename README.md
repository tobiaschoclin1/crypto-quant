# 🚀 Crypto Quant Pro

> **Sistema avanzado de trading cuantitativo para criptomonedas con análisis técnico en tiempo real**

---

## 📋 Descripción

**Crypto Quant Pro** es una plataforma completa de trading automatizado que combina análisis técnico avanzado, gestión de portfolio y alertas en tiempo real para operar en mercados de criptomonedas.

### ✨ Características Principales

- 📊 **Análisis técnico automatizado** con indicadores EMA50, MACD y RSI
- 💼 **Gestión de portfolio** multi-moneda (BTC, ETH, SOL, BNB, ADA)
- 🎯 **Sistema de señales inteligente** (Compra/Venta/Mantener)
- 🔄 **Sincronización en tiempo real** con datos de mercado
- 📱 **Interfaz web responsive** y app móvil nativa
- 🛡️ **Gestión de riesgo** con Stop Loss (3%) y Take Profit (4%)

---

## 🏗️ Arquitectura del Proyecto

```
📦 Project Cripto
├── 🔥 backend_api.py       # API FastAPI (Motor principal)
├── 🌐 index.html           # Interfaz web responsive
├── 📱 app_movil.py         # App móvil (Flet)
└── 📄 requirements.txt     # Dependencias Python
```

---

## 🚀 Instalación Rápida

### 1️⃣ **Clonar o descargar el proyecto**

```bash
cd "Project Cripto"
```

### 2️⃣ **Instalar dependencias**

```bash
pip install -r requirements.txt
```

### 3️⃣ **Iniciar el servidor**

```bash
python backend_api.py
```

El servidor se ejecutará en: `http://localhost:8000`

---

## 💻 Uso

### 🌐 **Interfaz Web**

1. Abre tu navegador en `http://localhost:8000`
2. Configura tu balance inicial en USDT
3. Observa las señales de trading en tiempo real
4. Registra operaciones manualmente o automáticamente

### 📱 **App Móvil**

Para usar la app móvil en tu dispositivo:

1. **Configurar IP local** en [app_movil.py](app_movil.py#L7):
   ```python
   API_URL = "http://TU_IP_LOCAL:8000/analisis"
   ```

2. **Ejecutar la app**:
   ```bash
   python app_movil.py
   ```

---

## 📊 Monedas Soportadas

| Símbolo | Nombre | Par |
|---------|---------|-----|
| 🟠 BTC  | Bitcoin | BTCUSDT |
| 🔷 ETH  | Ethereum | ETHUSDT |
| 🟣 SOL  | Solana | SOLUSDT |
| 🟡 BNB  | Binance Coin | BNBUSDT |
| 🔵 ADA  | Cardano | ADAUSDT |

---

## 🎯 Estrategia de Trading

### **Swing Trading Diario**

```
📈 Compra cuando:
   ✅ Precio > EMA50 (tendencia alcista)
   ✅ MACD > Señal (momentum positivo)
   ✅ RSI < 65 (no sobre-comprado)

📉 Venta cuando:
   ❌ Stop Loss: -3.0% (protección)
   ✅ Take Profit: +4.0% (objetivo)
   ⚠️  Precio < EMA50 (cambio de tendencia)
```

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Uso |
|-----------|-----|
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | API Backend |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Lenguaje principal |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Análisis de datos |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Cálculos numéricos |
| ![yFinance](https://img.shields.io/badge/yFinance-CC0000?style=flat&logo=yahoo&logoColor=white) | Datos de mercado |
| ![Flet](https://img.shields.io/badge/Flet-02569B?style=flat&logo=flutter&logoColor=white) | App móvil multiplataforma |

---

## 📡 Endpoints API

### `GET /analisis`
Obtiene análisis técnico y señales de trading

**Parámetros:**
- `symbol`: Par de trading (default: BTCUSDT)
- `current_price`: Precio actual (opcional)

**Respuesta:**
```json
{
  "symbol": "BTCUSDT",
  "precio": 67500.50,
  "decision": "COMPRA",
  "detalles": ["Entrada Swing Confirmada"],
  "portfolio": {...},
  "update_time": "14:35:22"
}
```

### `POST /set_balance`
Configura el balance inicial en USDT

### `POST /registrar_trade`
Registra una operación de compra o venta

### `GET /history/{symbol}`
Obtiene el historial de trades de una moneda

---

## 📱 Capturas de Pantalla

### Interfaz Web
```
┌─────────────────────────────────────────┐
│  💼 Crypto Quant Pro                    │
├─────────────────────────────────────────┤
│  Saldo: $10,000 USDT                    │
│                                          │
│  ┌──────────┐  ┌──────────┐            │
│  │ BTC/USDT │  │ ETH/USDT │            │
│  │ $67,500  │  │ $3,420   │            │
│  │ 📈 COMPRA │  │ ⏸ NEUTRAL│            │
│  └──────────┘  └──────────┘            │
│                                          │
│  📊 Análisis en Tiempo Real              │
│  EMA50: Alcista | MACD: Positivo        │
│  RSI: 58 (Zona Óptima)                  │
└─────────────────────────────────────────┘
```

---

## ⚙️ Configuración Avanzada

### Personalizar parámetros de trading

Edita [backend_api.py](backend_api.py#L24-L26):

```python
STOP_LOSS_PCT = 0.030      # 3.0% pérdida máxima
TAKE_PROFIT_PCT = 0.040    # 4.0% ganancia objetivo
```

### Agregar más criptomonedas

Modifica [backend_api.py](backend_api.py#L19):

```python
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]
```

---

## 🚨 Advertencias

⚠️ **Este software es para propósitos educativos y de investigación**

- No garantiza ganancias en operaciones reales
- Los mercados cripto son altamente volátiles
- Siempre realiza tu propia investigación (DYOR)
- Nunca inviertas más de lo que puedes permitirte perder

---

## 🤝 Contribuciones

¿Tienes ideas para mejorar el proyecto? ¡Las contribuciones son bienvenidas!

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo licencia libre. Puedes usarlo, modificarlo y distribuirlo libremente.

---

## 📞 Soporte

¿Necesitas ayuda? Abre un **Issue** en el repositorio o contacta al equipo de desarrollo.

---

<div align="center">

**Hecho con ❤️ para traders cuantitativos**

⭐ Si te gusta el proyecto, dale una estrella!

</div>
