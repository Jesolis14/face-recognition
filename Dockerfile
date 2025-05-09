FROM python:3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 💡 Reinstalación limpia desde fuente para asegurar numpy._core
RUN pip install --no-binary=numpy numpy==1.24.3

EXPOSE 8080

CMD ["python", "app.py"]
