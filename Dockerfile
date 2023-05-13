FROM python:3

# install wget and unzip
RUN apt-get update && apt-get install -y wget unzip

WORKDIR /app

# Copy the requirements.txt file.
COPY requirements.txt .

# Install the required Python packages.
RUN pip3 install -r requirements.txt

# Download and extract the models.zip file.
RUN wget https://github.com/maksymalist/JunkJudge/releases/download/v1.0/models.zip -O models.zip && \
    unzip -d /app models.zip && \
    rm models.zip

# Copy the current directory contents into the container at /app
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=main.py

# Start the Flask application.
CMD ["python", "main.py"]