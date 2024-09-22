# Use the official PyTorch image as a base
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Command to run the training script
CMD ["python", "src/train.py"]