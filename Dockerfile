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
# build-essential, curl, software-properties-common, git might be needed
# depending on your specific dependencies (e.g., if libraries require compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy application files (requirements.txt first to leverage Docker cache)
COPY requirements.txt ./
# Copy the .env file (optional, if used for API keys or config)
# Make sure your .env file is in the same directory as your Dockerfile locally
COPY .env ./
# Copy your application source code
COPY src/ ./src/

# Install Python dependencies
# --no-cache-dir reduces image size but might be slower on repeat builds without cache
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Permission and User Setup ---

# Create a non-root user and group named 'appuser'
RUN groupadd appuser && useradd -g appuser -m appuser

# Set STREAMLIT_HOME to a known writable location inside WORKDIR for the non-root user.
ENV STREAMLIT_HOME="/app/.streamlit"

# Explicitly create directories required by the application and Streamlit cache *as root*.
# This ensures they exist before switching users and can be correctly chowned.
# Include .streamlit (for config/cache), img (for sample images), and pdf_pages (for PDF extractions)
RUN mkdir -p "${STREAMLIT_HOME}" /app/img /app/pdf_pages

# Create the Streamlit config file to disable usage statistics (which caused the /.streamlit PermissionError).
# This file must be in the STREAMLIT_HOME directory. Created as root, will be chowned below.
RUN echo '[browser]' > "${STREAMLIT_HOME}/config.toml" && \
    echo 'gatherUsageStats = false' >> "${STREAMLIT_HOME}/config.toml"

# Change ownership of the /app directory (and its contents, including the directories/config file just created)
# to the non-root user. This ensures the non-root user can read/write here.
RUN chown -R appuser:appuser /app

# Switch to the non-root user for all subsequent commands and the entrypoint.
# The application code will now run as this user.
USER appuser

# --- Application Execution ---

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Healthcheck to verify the application is running and reachable on its port
# Adjust the endpoint if your app doesn't respond to /_stcore/health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit application as the non-root user
# --server.port=8501 is the port exposed and listened on within the container
# --server.address=0.0.0.0 binds to all network interfaces within the container
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]