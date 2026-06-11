FROM python:3.11.9

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Create output directories
RUN mkdir -p outputs/models outputs/figures outputs/predictions

# Run data loader then training
CMD ["sh", "-c", "python src/data_loader.py && python src/train/train.py"]