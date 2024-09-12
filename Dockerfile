# Use Python base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
