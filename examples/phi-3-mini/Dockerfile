# Dockerfile for Phi-3-mini with vLLM

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the example code
COPY examples/phi-3-mini/ ./examples/phi-3-mini/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Expose the port that vLLM will run on
EXPOSE 8000

# Set the default command to serve the Phi-3-mini model
# Note: The model will be downloaded on first run, or you can mount a volume with the model
CMD ["bash", "-c", "cd examples/phi-3-mini && ./serve.sh"]