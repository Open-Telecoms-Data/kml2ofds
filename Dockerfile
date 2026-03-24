# kml2ofds web service
FROM python:3.10-slim

WORKDIR /app

# Install build deps for geopandas/shapely
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY web/ web/

# Install with web extra
RUN pip install --no-cache-dir -e ".[web]"

EXPOSE 8000

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
