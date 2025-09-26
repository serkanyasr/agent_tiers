FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ca-certificates \
    gnupg \
    lsb-release \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN pip install uv && uv pip install --system -r pyproject.toml

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8050 8055 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "src.agent_tiers.infrastructure.api.main"]
