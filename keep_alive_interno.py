"""
KEEP-ALIVE INTERNO (OPCIONAL)
==============================
Este archivo es opcional. Solo úsalo si NO quieres usar servicios externos como UptimeRobot.

Para activarlo:
1. Instala la dependencia: pip install apscheduler aiohttp
2. Agrega al requirements.txt: apscheduler==3.10.4
3. Importa este módulo en backend_api.py: from keep_alive_interno import start_keep_alive
4. Llama start_keep_alive() después de crear la app en backend_api.py
"""

import asyncio
import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
import os

RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # Render provee esta variable automáticamente

async def ping_self():
    """Hace ping al propio endpoint de health para mantener la app activa"""
    if not RENDER_URL:
        print("[Keep-Alive] RENDER_EXTERNAL_URL no configurada, skip ping")
        return

    try:
        url = f"{RENDER_URL}/health"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    print(f"[Keep-Alive] Ping exitoso a {url} - {datetime.now().strftime('%H:%M:%S')}")
                else:
                    print(f"[Keep-Alive] Ping falló con status {response.status}")
    except Exception as e:
        print(f"[Keep-Alive] Error en ping: {e}")

def start_keep_alive():
    """Inicia el scheduler para hacer ping cada 10 minutos"""
    scheduler = AsyncIOScheduler()

    # Ping cada 10 minutos (600 segundos)
    scheduler.add_job(ping_self, 'interval', minutes=10, id='keep_alive_ping')

    scheduler.start()
    print("[Keep-Alive] Scheduler iniciado - ping cada 10 minutos")

    return scheduler
