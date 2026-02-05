FROM python:3.11

WORKDIR /app
RUN echo "### USING BACKEND DOCKERFILE ###"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose default port
ENV PORT=5000

# Use shell-form CMD so $PORT expands at runtime
CMD exec gunicorn app:app \
    --workers 2 \
    --threads 4 \
    --timeout 300 \
    --preload \
    --bind 0.0.0.0:$PORT \
    --access-logfile - \
    --error-logfile - \