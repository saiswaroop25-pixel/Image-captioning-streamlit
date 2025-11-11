FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	HF_HOME=/root/.cache/huggingface \
	TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
	TORCH_HOME=/root/.cache/torch

WORKDIR /app

# System deps for Pillow/torch and image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	curl \
	libgl1 \
	libglib2.0-0 \
	libjpeg62-turbo \
	zlib1g \
	git \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Prefetch model weights during build is optional (can be slow on small EC2)
# Enable with: docker build --build-arg PREFETCH=true -t image-captioning:latest .
ARG PREFETCH=false
RUN if [ "$PREFETCH" = "true" ]; then python bootstrap_models.py || true; else echo "Skipping model prefetch"; fi

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


