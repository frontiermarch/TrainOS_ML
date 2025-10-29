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

# Expose port for Render
EXPOSE 10000

# Command to run your app
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:10000"]
