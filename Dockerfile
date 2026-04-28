# Base image — official Python 3.9 slim variant
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker layer caching — explained below)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command — run pytest to verify everything works
CMD ["pytest", "tests/", "-v"]