FROM python:3.11                 # Use the official Python 3.11 image as the base image

COPY . /app                      # Copy the current directory contents into the /app directory inside the container

WORKDIR /app                     # Set the working directory to /app inside the container

RUN pip install -r requirements.txt  # Install dependencies from the requirements.txt file

EXPOSE 8501                       # Expose port 8501, the default Streamlit port

CMD ["streamlit", "run", "app.py", "--workers=4", "--server.address=0.0.0.0", "--server.port=8501"]  # Start Streamlit app
