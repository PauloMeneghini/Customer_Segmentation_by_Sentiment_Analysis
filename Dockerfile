FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Diretório onde o DuckDB vai persistir os dados
RUN mkdir -p /data
VOLUME ["/data"]

# Caminho padrão do banco (pode ser sobrescrito via -e no docker run)
ENV DB_PATH=/data/olist_analytics.db

CMD ["python", "main.py"]