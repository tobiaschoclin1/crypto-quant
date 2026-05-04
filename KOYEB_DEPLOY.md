# 🚀 Guía de Deploy en Koyeb - Crypto Quant Pro

## ✨ Por qué Koyeb

- ✅ **Gratis**: Plan gratuito generoso
- ✅ **No se duerme**: A diferencia de Render, está siempre activo
- ✅ **512MB RAM**: Suficiente para análisis técnico
- ✅ **Deploy automático**: Conecta con GitHub
- ✅ **SSL gratis**: HTTPS automático

---

## 📋 Requisitos Previos

- ✅ Cuenta en Koyeb (gratis): [koyeb.com](https://www.koyeb.com)
- ✅ Repositorio en GitHub
- ✅ Proyecto con Dockerfile configurado (✅ ya está)

---

## 🚀 Paso a Paso

### 1️⃣ Crear Cuenta en Koyeb

1. Ve a [koyeb.com](https://www.koyeb.com)
2. Click en **"Sign up"**
3. Conecta con GitHub (recomendado)
4. Completa el registro (no requiere tarjeta de crédito)

---

### 2️⃣ Crear Nueva App

1. En el Dashboard de Koyeb, click en **"Create App"**
2. Selecciona **"GitHub"** como source
3. Autoriza a Koyeb para acceder a tus repositorios (si es la primera vez)
4. Busca y selecciona tu repositorio: **`crypto-quant`** o como se llame

---

### 3️⃣ Configurar el Deployment

**Builder:**
- Selecciona **"Dockerfile"** (Koyeb detectará automáticamente el Dockerfile)

**Instance:**
- **Type**: Web
- **Regions**: Elegir la más cercana (ej: Paris, Frankfurt, o Washington)
- **Instance type**: **"Eco"** (el plan gratuito - 512MB RAM)

**Port:**
- **Port**: `8000` (el puerto que usa tu app)

**Health checks:**
- **HTTP Path**: `/health` (ya tienes este endpoint en tu código)
- Dejar los demás valores por defecto

---

### 4️⃣ Variables de Entorno (Opcional)

Si tienes variables de entorno, agrégalas aquí.

Para este proyecto **NO son necesarias** (todo está en el código).

---

### 5️⃣ Naming & Deploy

1. **App name**: `crypto-quant-pro` (o el nombre que prefieras)
2. **Service name**: `api` (o dejar por defecto)
3. Click en **"Deploy"**

⏱️ **El deploy tomará 3-5 minutos**

---

## 🌐 Obtener tu URL

Una vez completado el deploy:

1. Ve a tu app en el Dashboard
2. Encontrarás una URL como:
   ```
   https://crypto-quant-pro-XXXXX.koyeb.app
   ```
3. Copia esta URL

---

## ✅ Verificar que Funciona

1. Visita: `https://tu-app.koyeb.app`
2. Deberías ver tu interfaz web de Crypto Quant Pro
3. Prueba los análisis técnicos
4. Verifica que los datos se cargan correctamente

**Endpoints para probar:**
- `https://tu-app.koyeb.app/` → Interfaz web
- `https://tu-app.koyeb.app/health` → Health check
- `https://tu-app.koyeb.app/analisis?symbol=BTCUSDT` → Análisis de Bitcoin

---

## 🔄 Redeploy Automático

Koyeb hace redeploy automáticamente cuando:
- Haces push a tu rama principal en GitHub
- Cambias variables de entorno
- Lo haces manualmente desde el dashboard

---

## 📊 Monitoreo

**Dashboard de Koyeb muestra:**
- CPU usage
- Memory usage
- Request count
- Response times
- Logs en tiempo real

---

## 🛠️ Troubleshooting

### Build falla

**Error común: "No Dockerfile found"**
- Verifica que el Dockerfile esté en la raíz del repo
- Asegúrate de que se haya pusheado a GitHub

**Error: Dependencias no se instalan**
- Verifica que `requirements.txt` esté actualizado
- Revisa los logs de build en Koyeb

### App no responde

**Error 502 Bad Gateway**
- Verifica que el puerto sea `8000`
- Revisa logs en Koyeb Dashboard
- Asegúrate de que `backend_api.py` esté configurado para usar `PORT` env var

### Out of Memory

Si ves errores de memoria:
- El plan Eco tiene 512MB RAM
- Verifica que no estés cargando demasiados datos en memoria
- Optimiza el código si es necesario

---

## 🎯 Diferencias con Render

| Feature | Koyeb | Render |
|---------|-------|--------|
| **Se duerme** | ❌ NO | ✅ SÍ (15 min) |
| **RAM Gratis** | 512MB | 512MB |
| **Velocidad** | 🚀 Rápido | ⚡ Medio |
| **UptimeRobot** | ❌ No necesario | ✅ Necesario |
| **SSL** | ✅ Automático | ✅ Automático |

---

## 🧹 Limpieza de Render

Una vez que Koyeb funcione correctamente:

1. Ve a [dashboard.render.com](https://dashboard.render.com)
2. Busca `crypto-quant-pro` (o el nombre de tu servicio)
3. Settings → **Delete Service**

---

## 📱 UptimeRobot

**¿Necesitas UptimeRobot?**

❌ **NO** - Koyeb no se duerme

**Opcional:** Puedes configurarlo solo para recibir alertas si la app cae.

---

## 💰 Costos

**Plan Eco (Gratuito):**
- ✅ 1 app web
- ✅ 512MB RAM
- ✅ Sin tarjeta de crédito
- ✅ Sin límite de tiempo

**Si necesitas más recursos:**
- Planes desde $7/mes (pero no debería ser necesario)

---

## 🔗 URLs Importantes

- Dashboard: [app.koyeb.com](https://app.koyeb.com)
- Docs: [koyeb.com/docs](https://www.koyeb.com/docs)
- Status: [status.koyeb.com](https://status.koyeb.com)

---

## 📋 Checklist Final

```
[ ] 1. Crear cuenta en Koyeb
[ ] 2. Conectar repositorio de GitHub
[ ] 3. Configurar deployment (Dockerfile, port 8000)
[ ] 4. Deploy
[ ] 5. Obtener URL de Koyeb
[ ] 6. Verificar que funcione
[ ] 7. Eliminar servicio de Render
[ ] 8. (Opcional) Eliminar UptimeRobot
```

---

## 🎉 ¡Listo!

Tu app estará **100% operativa, sin dormirse, y gratis** en Koyeb.

**URL final**: `https://crypto-quant-pro-XXXXX.koyeb.app`

---

**🚀 ¡Empecemos! Ve a [koyeb.com](https://www.koyeb.com) y crea tu cuenta.**
