FROM python:3.11

WORKDIR /app
RUN echo "### USING BACKEND DOCKERFILE ###"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab

# Copy app
COPY . .

# Expose default port
ENV PORT=5000
EXPOSE 5000

# Use shell-form CMD so $PORT expands at runtime
CMD gunicorn app:app --workers 2 --bind 0.0.0.0:${PORT:-5000}
