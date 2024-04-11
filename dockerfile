# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install vectorbt and any other dependencies
RUN pip install --no-cache-dir numpy pandas vectorbt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Command to run the backtesting script
CMD ["python", "./vbt_test.py"]
