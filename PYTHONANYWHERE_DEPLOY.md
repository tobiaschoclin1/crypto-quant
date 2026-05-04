# 🐍 Guía de Deploy en PythonAnywhere - Crypto Quant Pro

## ✨ Por qué PythonAnywhere

- ✅ **100% Gratis**: Sin tarjeta de crédito
- ✅ **No se duerme**: Siempre activo
- ✅ **Especializado en Python**: Optimizado para apps Python
- ✅ **512MB storage**: Suficiente para el proyecto
- ✅ **Fácil de usar**: Consola web incluida

## ⚠️ Limitaciones del Free Tier

- ⚠️ **HTTP only**: No HTTPS personalizado (pero pythonanywhere.com tiene SSL)
- ⚠️ **URL**: `username.pythonanywhere.com`
- ⚠️ **CPU limitado**: 100 segundos/día de CPU
- ⚠️ **APIs externas**: Solo whitelist (yfinance está permitido)

---

## 📋 Paso a Paso

### 1️⃣ Crear Cuenta en PythonAnywhere

1. Ve a: [pythonanywhere.com](https://www.pythonanywhere.com)
2. Click en **"Start running Python online in less than a minute!"**
3. Click en **"Create a Beginner account"** (gratis)
4. Completa el registro:
   - Username (importante: esto será tu URL)
   - Email
   - Password
5. ✅ Confirma tu email

**Tu URL será:** `https://TU_USERNAME.pythonanywhere.com`

---

### 2️⃣ Subir tu Código

**Opción A: Desde GitHub (Recomendado)**

1. En PythonAnywhere, ve a **"Consoles"** → **"Bash"**
2. En la consola, ejecuta:

```bash
# Clonar tu repositorio
git clone https://github.com/TU_USUARIO/crypto-quant.git

# Renombrar a crypto-quant-pro (más simple)
mv crypto-quant crypto-quant-pro

# Entrar al directorio
cd crypto-quant-pro
```

**Opción B: Upload Manual**

1. Ve a **"Files"**
2. Crea un directorio: `crypto-quant-pro`
3. Sube los archivos:
   - `backend_api.py`
   - `index.html`
   - `requirements-pythonanywhere.txt`
   - `wsgi.py`

---

### 3️⃣ Instalar Dependencias

En la consola Bash de PythonAnywhere:

```bash
cd crypto-quant-pro

# Instalar dependencias
pip3.10 install --user -r requirements-pythonanywhere.txt
```

⏱️ **Esto tomará 3-5 minutos**

---

### 4️⃣ Configurar Web App

1. Ve a **"Web"** en el menú superior
2. Click en **"Add a new web app"**
3. **Domain**: Acepta `username.pythonanywhere.com`
4. **Framework**: Selecciona **"Manual configuration"**
5. **Python version**: Selecciona **"Python 3.10"**
6. Click **"Next"**

---

### 5️⃣ Configurar WSGI File

1. En la página de configuración de tu web app, busca la sección **"Code"**
2. Click en el link del **WSGI configuration file** (ej: `/var/www/username_pythonanywhere_com_wsgi.py`)
3. **Borra todo el contenido** del archivo
4. **Copia y pega** este código (reemplaza `TU_USERNAME`):

```python
import sys
import os

# Agregar el directorio del proyecto al path
project_home = '/home/TU_USERNAME/crypto-quant-pro'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Importar la app FastAPI
from backend_api import app

# PythonAnywhere necesita una aplicación WSGI
from asgiref.wsgi import WsgiToAsgi

application = WsgiToAsgi(app)
```

5. Click en **"Save"** (arriba a la derecha)

---

### 6️⃣ Configurar Static Files (Para index.html)

En la página de configuración de tu web app:

1. Busca la sección **"Static files"**
2. Click en **"Enter URL"**
   - **URL**: `/`
   - **Directory**: `/home/TU_USERNAME/crypto-quant-pro`
3. Click en ✅ (check verde)

---

### 7️⃣ Configurar Variables de Entorno (Opcional)

Si necesitas variables de entorno:

1. Edita el archivo WSGI
2. Agrega al inicio (antes de importar app):

```python
os.environ['PORT'] = '8000'
# Otras variables aquí
```

---

### 8️⃣ Reload y Verificar

1. En la página de configuración, click en el botón verde **"Reload"** (arriba a la derecha)
2. Espera 10-15 segundos
3. Visita tu URL: `https://TU_USERNAME.pythonanywhere.com`

**Endpoints para verificar:**
- `https://TU_USERNAME.pythonanywhere.com/` → Interfaz web
- `https://TU_USERNAME.pythonanywhere.com/health` → Health check
- `https://TU_USERNAME.pythonanywhere.com/analisis?symbol=BTCUSDT` → Análisis BTC

---

## 🛠️ Troubleshooting

### Error 502 Bad Gateway

**Causa común**: Error en el código o dependencias no instaladas

**Solución:**
1. Ve a **"Web"** → Sección **"Log files"**
2. Click en **"Error log"**
3. Lee el error y corrígelo
4. Haz **"Reload"** de nuevo

### Error: Module not found

**Causa**: Dependencia no instalada

**Solución:**
```bash
cd crypto-quant-pro
pip3.10 install --user nombre_del_modulo
```
Luego **"Reload"** en Web

### Error: yfinance no puede conectar

**Causa**: PythonAnywhere solo permite APIs de su whitelist

**Solución:** yfinance debería estar permitido. Si no funciona:
1. Ve a **"Account"** → **"API access"**
2. Verifica que yfinance esté en la lista permitida
3. Si es necesario, usa datos cacheados en el código

### App muy lenta

**Causa**: Límite de CPU del free tier

**Solución:**
- Optimiza consultas a APIs
- Cachea datos cuando sea posible
- Reduce la frecuencia de updates

---

## 📊 Monitoreo

**Dashboard de PythonAnywhere:**
- CPU usage (límite: 100 seg/día)
- Web traffic
- Error logs
- Access logs

---

## 🔄 Actualizar el Código

**Opción A: Git pull**

```bash
cd crypto-quant-pro
git pull origin main
```

Luego **"Reload"** en Web

**Opción B: Upload manual**

1. Ve a **"Files"**
2. Sube los archivos modificados
3. **"Reload"** en Web

---

## 🎯 Comparativa con Otras Plataformas

| Feature | PythonAnywhere | Railway | Render |
|---------|---------------|---------|--------|
| **Tarjeta** | ❌ NO | ❌ NO | ❌ NO |
| **Se duerme** | ❌ NO | ❌ NO | ✅ SÍ |
| **HTTPS** | ✅ SÍ | ✅ SÍ | ✅ SÍ |
| **Custom Domain** | ❌ NO (free) | ✅ SÍ | ✅ SÍ |
| **CPU Limit** | ⚠️ 100s/día | ✅ Generoso | ✅ OK |

---

## 💡 Tips

- **Usa caching**: Para reducir llamadas a APIs externas
- **Optimiza consultas**: Para no exceder el límite de CPU
- **Logs son tu amigo**: Revisa los logs si algo falla
- **Scheduled tasks**: Disponibles en el free tier (1/día)

---

## 🧹 Limpieza de Render

Una vez que PythonAnywhere funcione:

1. Ve a [dashboard.render.com](https://dashboard.render.com)
2. Elimina el servicio `crypto-quant-pro`
3. Elimina UptimeRobot (ya no lo necesitas)

---

## 📋 Checklist Final

```
[ ] 1. Crear cuenta en PythonAnywhere
[ ] 2. Clonar repositorio o subir archivos
[ ] 3. Instalar dependencias
[ ] 4. Crear Web App (Manual configuration)
[ ] 5. Configurar WSGI file
[ ] 6. Configurar static files (opcional)
[ ] 7. Reload y verificar
[ ] 8. Probar todos los endpoints
[ ] 9. Eliminar servicio de Render
[ ] 10. Eliminar UptimeRobot
```

---

## 🔗 URLs Importantes

- Dashboard: [pythonanywhere.com/user/TU_USERNAME](https://www.pythonanywhere.com)
- Docs: [help.pythonanywhere.com](https://help.pythonanywhere.com)
- Forums: [pythonanywhere.com/forums](https://www.pythonanywhere.com/forums)

---

## 🎉 ¡Listo!

Tu app estará **100% gratis, sin dormirse** en PythonAnywhere.

**URL final**: `https://TU_USERNAME.pythonanywhere.com`

---

## ⚠️ Importante: Antes de Empezar

**Necesitas hacer push a GitHub del archivo `wsgi.py`:**

```bash
cd "Project Cripto"
git add wsgi.py requirements-pythonanywhere.txt PYTHONANYWHERE_DEPLOY.md
git commit -m "Add PythonAnywhere configuration"
git push origin main
```

Luego ya puedes clonar el repo en PythonAnywhere.

---

**🚀 ¡Empecemos! Ve a [pythonanywhere.com](https://www.pythonanywhere.com) y crea tu cuenta.**
