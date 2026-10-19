# 1. Use an official, lightweight Python slim runtime base image architecture
FROM python:3.12-slim

# 2. Establish an internal container working directory zone
WORKDIR /app

# 3. Copy system requirements manifest file directly inside image layer
COPY requirements.txt .

# 4. Install all application package infrastructure builds cleanly without caches
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy all project application and generated model directory assets inside
COPY src/ ./src/
COPY mlruns/ ./mlruns/

# 6. Expose the structural network routing port mapping used by FastAPI
EXPOSE 8000

# 7. Start the production async server gateway launcher framework on execution activation
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
