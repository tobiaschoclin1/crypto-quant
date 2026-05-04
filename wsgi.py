# WSGI configuration file for PythonAnywhere
# Este archivo permite que FastAPI funcione en PythonAnywhere

import sys
import os

# Agregar el directorio del proyecto al path
project_home = '/home/TU_USERNAME/crypto-quant-pro'  # Cambiar TU_USERNAME
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Importar la app FastAPI
from backend_api import app

# PythonAnywhere necesita una aplicación WSGI
# FastAPI es ASGI, pero podemos usar un adapter
from asgiref.wsgi import WsgiToAsgi

application = WsgiToAsgi(app)
