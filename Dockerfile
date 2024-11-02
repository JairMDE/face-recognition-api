# Use an official Python image as a base
FROM python:3.10-slim

# Set the working directory to the root directory of the container
WORKDIR /

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other files and folders (like server.py and models)
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Command to start the application with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "server:app"]
