# Instalación: Keep-Alive Interno (Opcional)

**⚠️ NOTA**: Esta opción es SOLO si NO quieres usar servicios externos como UptimeRobot o Cron-job.org.

La opción externa (UptimeRobot) es más recomendada porque:
- Es gratis
- No consume recursos de tu app
- Más confiable
- Te avisa si la app cae

## Si aún así prefieres la opción interna:

### 1. Actualiza requirements.txt

Agrega esta línea al archivo `requirements.txt`:
```
apscheduler==3.10.4
```

### 2. Modifica backend_api.py

Agrega estos imports al inicio del archivo (después de las otras importaciones):

```python
from keep_alive_interno import start_keep_alive
```

Luego, al final del archivo (antes de `if __name__ == "__main__":`), agrega:

```python
# Iniciar keep-alive interno
start_keep_alive()
```

### 3. Configura variable de entorno en Render

En tu dashboard de Render:
1. Ve a tu servicio
2. Entra a "Environment"
3. Verifica que existe la variable `RENDER_EXTERNAL_URL` (Render la crea automáticamente)

### 4. Redeploy

```bash
git add .
git commit -m "Add: Keep-alive interno para evitar sleep"
git push
```

## Cómo funciona

- La app se hace ping a sí misma cada 10 minutos
- Usa el endpoint `/health` que ya creamos
- Se ejecuta en background sin bloquear otras operaciones

## Desventajas vs servicio externo

- ❌ Consume recursos de tu propia app
- ❌ Si la app crashea, el keep-alive también cae
- ❌ Más complejo de debuggear
- ❌ Una dependencia adicional

## ¿Cuál usar?

**Recomendado**: UptimeRobot (ver KEEP_ALIVE.md)
**Solo si necesitas**: Opción interna (estas instrucciones)
