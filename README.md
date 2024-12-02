# Setup Instructions

## Dev environment setup
```bash
# Check python version
python3 --version 
> Python 3.11.10

# Activate python environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python3 stream_processor.py
```

## Other commands / reference / notes
```bash


pip install opencv-python-headless ultralytics torch torchvision



# Create a virtual environment, this creates a virtual environment in a folder named env.
python -m venv env

# Activate virtual environment
source env/bin/activate

# Deactivate virtual environment
deactivate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Add/install dependency
pip install package_name
pip install package_name==version_number
```