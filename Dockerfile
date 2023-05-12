# Use the official Amazon Linux 2 image.
# https://hub.docker.com/_/amazonlinux
FROM amazonlinux:2

# Install the required packages to build and run Python applications.
RUN yum -y update && \
    yum -y install python3 python3-devel python3-pip wget unzip && \
    yum clean all

# Set the working directory.
WORKDIR /app

# Copy the requirements.txt file.
COPY requirements.txt .

# Install the required Python packages.
RUN pip3 install --no-cache-dir -r requirements.txt

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

# Start the Flask application.
CMD ["flask", "run", "--host=0.0.0.0"]
