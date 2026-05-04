# Dockerfile optimizado para Koyeb
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY backend_api.py .
COPY index.html .

# Exponer el puerto (Koyeb usa PORT dinámico)
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "backend_api.py"]
