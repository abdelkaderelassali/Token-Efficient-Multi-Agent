FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We also need uvicorn to run the FastAPI app
RUN pip install --no-cache-dir uvicorn fastapi

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
