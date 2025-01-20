# -------------------------------------------------------
# 1) Base stage: install dependencies + download models
# -------------------------------------------------------
    FROM python:3.10-slim as base

    # Disable .pyc and enable unbuffered Python logs
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    # Copy just requirements + the model-download script (for Docker caching)
    COPY requirements.txt .
    COPY download_models_hf.py .
    
    # Install system packages needed by Paddle/OpenCV
    RUN apt-get update && \
        apt-get install --yes --no-install-recommends \
        curl \
        g++ \
        libgomp1 \
        libopencv-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # Install dependencies
    RUN pip install --no-cache-dir --upgrade pip \
        && pip install --no-cache-dir -r requirements.txt --extra-index-url https://wheels.myhloli.com \
        && pip install --no-cache-dir -U "magic-pdf[full]" huggingface_hub --extra-index-url https://wheels.myhloli.com
    
    # Download models with retry logic
    RUN python download_models_hf.py || (echo "Retrying model download..." && python download_models_hf.py)
    
    # -------------------------------------------------------
    # 2) Development stage: local code volume, run manage.py
    # -------------------------------------------------------
    FROM base as development
    
    EXPOSE 5000
    
    # In dev, rely on Docker Compose volumes for live code updates
    CMD ["python", "manage.py", "runserver", "0.0.0.0:5000"]
    
    # -------------------------------------------------------
    # 3) Production stage: copy code and use gunicorn
    # -------------------------------------------------------
    FROM base as production
    
    # Copy the entire extractor_api folder into /app
    COPY extractor_api /app
    
    # Install gunicorn
    RUN pip install --no-cache-dir gunicorn whitenoise
    
    EXPOSE 5000
    
    # Reference "api.wsgi:application" for production
    CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "3000", "api.wsgi:application"]
    