FROM python:3.10-slim

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY . /app

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --force-reinstall numpy==1.24.3
# Exponer el puerto (usado por Flask)
EXPOSE 8000

# Comando de arranque
CMD ["python", "app.py"]
