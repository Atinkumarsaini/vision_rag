# FROM python:3.13.3

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt ./
# COPY src/ ./src/

# RUN pip3 install -r requirements.txt

# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]



FROM python:3.13.3

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy application files
# requirements.txt first to leverage Docker cache
COPY requirements.txt ./
# Copy the .env file before installing requirements (optional, but ensures it's there)
# Make sure your .env file is in the same directory as your Dockerfile locally
COPY .env ./
# Copy source code
COPY src/ ./src/

# Install Python dependencies
# pip3 install --no-cache-dir -r requirements.txt will now install python-dotenv
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Permission and User Setup (from previous answer, still important) ---

# Streamlit sometimes tries to write config/cache outside WORKDIR or as root.
# Set STREAMLIT_HOME to a known writable location inside WORKDIR.
ENV STREAMLIT_HOME="/app/.streamlit"
# Create the .streamlit directory. It will initially be owned by root.
RUN mkdir -p "${STREAMIT_HOME}"

# Create a non-root user and group named 'appuser'
RUN groupadd appuser && useradd -g appuser -m appuser

# Change ownership of the /app directory (and its contents) to the non-root user.
# This ensures the non-root user can read/write here.
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# --- Application Execution ---

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to verify the application is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit application as the non-root user
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]