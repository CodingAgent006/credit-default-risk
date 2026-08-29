FROM python:3.10-slim

WORKDIR /app

# 1. Optimize Python runtime in container environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy source files
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

# 4. Production start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]