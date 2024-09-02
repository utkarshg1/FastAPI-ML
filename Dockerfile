# Use an official Python runtime as a parent image
FROM python

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn (production-ready ASGI server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
