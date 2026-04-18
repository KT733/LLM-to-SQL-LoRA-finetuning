# Use Python 3.10 as the base image
FROM python:3.10

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]

# Update & install necessary utilities
RUN apt-get update -qq && apt-get upgrade -qq && \
    apt-get install -qq man wget sudo vim tmux

# Upgrade pip
RUN yes | pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the my_model folder and other application files into the container
COPY my_model /app/my_model
COPY main.py /app
COPY unit_test.py /app
COPY sample_output_hw3.txt /app

# Set the environment variable to prevent Python buffering (optional)
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "main.py"]
