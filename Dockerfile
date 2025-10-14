FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04


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

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

    
# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python
# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN pip install uv && uv pip install --system -r pyproject.toml

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8050 8055 8501 8502

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "src.agent_tiers.infrastructure.api.main"]
