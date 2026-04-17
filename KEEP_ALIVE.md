# Keep-Alive para Render

Este proyecto incluye un sistema para evitar que la aplicación entre en sleep en Render después de 15 minutos de inactividad.

## Endpoint de Health Check

Ya está implementado el endpoint: `GET /health`

Este endpoint devuelve:
```json
{
  "status": "ok",
  "timestamp": "2026-04-16T18:30:00.000Z"
}
```

## Opción 1: UptimeRobot (Recomendado - GRATIS)

1. Ve a [https://uptimerobot.com](https://uptimerobot.com) y crea una cuenta gratuita
2. Crea un nuevo monitor:
   - **Monitor Type**: HTTP(s)
   - **Friendly Name**: Crypto App Keep-Alive
   - **URL**: `https://tu-app.onrender.com/health`
   - **Monitoring Interval**: 5 minutos (plan gratuito permite cada 5 min)
3. Guarda el monitor

**Ventajas**: Gratis, sin código adicional, dashboard de monitoreo, alertas si tu app cae

## Opción 2: Cron-job.org (GRATIS)

1. Ve a [https://cron-job.org](https://cron-job.org) y crea una cuenta
2. Crea un nuevo cron job:
   - **Title**: Crypto Keep-Alive
   - **URL**: `https://tu-app.onrender.com/health`
   - **Schedule**: Cada 10 minutos (*/10 * * * *)
3. Activa el job

**Ventajas**: Gratis, flexible, configuración de horarios precisos

## Opción 3: GitHub Actions (GRATIS)

Si tu código está en GitHub, puedes usar Actions para hacer ping:

```yaml
# .github/workflows/keep-alive.yml
name: Keep-Alive
on:
  schedule:
    - cron: '*/10 * * * *'  # Cada 10 minutos
  workflow_dispatch:

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping health endpoint
        run: curl -f https://tu-app.onrender.com/health || exit 0
```

## Notas Importantes

- **Render Free Tier**: Se duerme después de 15 minutos de inactividad
- **Recomendación**: Hacer ping cada 10-14 minutos
- **No exagerar**: Hacer ping cada 1 minuto es innecesario y puede considerarse abuso
- **Alternativa definitiva**: Actualizar a plan de pago de Render ($7/mes) elimina el sleep

## Verificación

Para verificar que funciona:
1. Espera 16+ minutos sin usar tu app
2. Visita `https://tu-app.onrender.com/health`
3. Si responde rápido (sin pantalla de carga), el keep-alive está funcionando
