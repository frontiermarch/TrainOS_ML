# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose port (optional, for clarity)
EXPOSE 5000

# Command to run the app using Render's PORT env
CMD ["sh", "-c", "gunicorn app:app -b 0.0.0.0:${PORT:-5000} -w 4"]
