# Use a Python 3.11 image as base
FROM python:3.11

# Set the working directory to /app
WORKDIR /app

# Copy the content of the current directory to the /app directory in the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on (default is 8501)
EXPOSE 8501

# Set the command to run Streamlit
CMD ["streamlit", "run", "app.py"]
