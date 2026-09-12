# Use a stable, modern python image
From python:3.11-slim

# Set the working directory
WORKDIR /app

# Install the openai library
RUN pip install --no-cache-dir openai

#Coy the code from GitHub to the container
COPY . .

# Run the script
CMD ["python", "main_v2.py"]
