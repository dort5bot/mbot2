#Final-1:cache temizleme>Python cache'leri temizleniyor+
# Build aşaması
FROM python:3.11-slim AS builder

WORKDIR /app

# Build bağımlılıklarını kur (eksik paketler eklendi)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libc-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pip'i güncelle (derleme sorunlarını önlemek için)
RUN pip install --upgrade pip setuptools wheel

# Bağımlılıkları kopyala ve wheel olarak derle (--no-deps KALDIRILDI)
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# requirements.txt'yi de wheels klasörüne kopyala
RUN cp requirements.txt /app/wheels/

# Runtime aşaması
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime bağımlılıkları
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Uygulama kullanıcısı oluştur
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Python optimizasyonları
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp \
    PIP_NO_CACHE_DIR=1

# Wheel'ları ve requirements.txt'yi kopyala
COPY --from=builder /app/wheels /wheels

# Wheel'lardan paketleri kur + CACHE TEMİZLEME
RUN pip install --no-index --find-links=/wheels -r /wheels/requirements.txt \
    && rm -rf /wheels \
    && find /usr/local -depth \
      \( \
        \( -type d -name __pycache__ \) \
        -o \( -type f -name '*.pyc' \) \
        -o \( -type f -name '*.pyo' \) \
      \) -exec rm -rf {} +

# Uygulama kodunu kopyala
COPY --chown=appuser:appgroup . .

# Uygulama kodundaki cache'leri temizle
RUN find . -depth \
      \( \
        \( -type d -name __pycache__ \) \
        -o \( -type f -name '*.pyc' \) \
        -o \( -type f -name '*.pyo' \) \
      \) -exec rm -rf {} +

# Health check ve port ayarları
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Çalışma kullanıcısını ayarla
USER appuser

# Çalıştırma komutu
CMD ["python", "main.py"]
