FROM python:3.13-slim

WORKDIR /app

# Install system dependencies if any (none for now)

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Code
COPY src/ src/
COPY config.yaml .

# Default command (can be overridden)
CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
