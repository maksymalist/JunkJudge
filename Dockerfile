# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 5000 for the Flask app to listen on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=main.py

# Run the command to start the Flask server
CMD ["flask", "run", "--host=0.0.0.0"]