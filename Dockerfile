FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY reports/best_strategy_daily.json reports/best_strategy_daily.json
COPY run_pipeline.py .

# Data and models are mounted as volumes, never baked into the image.
VOLUME ["/app/data", "/app/models"]

# Daily update: download new data, rebuild the dataset, print/send today's book.
# Restore a copy first with:  docker run ... python -m src.hf_sync pull <user>/<repo>
CMD ["python", "run_pipeline.py"]
