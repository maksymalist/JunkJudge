# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim-buster

# Set the working directory.
WORKDIR /app

# Copy the requirements.txt file.
COPY requirements.txt .

# Install wget, unzip, and Flask.
RUN apt-get update && apt-get install -y wget unzip && pip install -r requirements.txt

# Download and extract the models.zip file.
RUN wget https://github.com/maksymalist/JunkJudge/releases/download/v1.0/models.zip -O models.zip && \
    unzip -d /app models.zip && \
    rm models.zip

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 5000 for the Flask app to listen on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=main.py

RUN  ls -l

# Start the Flask application.
CMD ["flask", "run", "--host=0.0.0.0"]
