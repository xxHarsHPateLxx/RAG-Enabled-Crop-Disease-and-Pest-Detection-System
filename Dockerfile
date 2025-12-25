FROM python:3.10-slim

WORKDIR /app

# System dependencies for FAISS & TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire server directory contents to /app
COPY server/ .

EXPOSE 7860

# Run from /app where main.py is located
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
