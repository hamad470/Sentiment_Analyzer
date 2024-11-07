# Dockerfile

# Use a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure NLTK data is available
RUN python -m nltk.downloader -d ./nltk_data punkt punkt_tab stopwords

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
