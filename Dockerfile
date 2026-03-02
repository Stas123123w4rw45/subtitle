FROM python:3.10-slim

# Install system dependencies (FFmpeg)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Create a non-root user with ID 1000 (standard for HF Spaces)
RUN useradd -m -u 1000 user

# Copy files
COPY --chown=user . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER user

# Run bot
RUN chmod +x start.sh
CMD ["./start.sh"]
