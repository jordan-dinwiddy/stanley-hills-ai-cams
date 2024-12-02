# Setup Instructions
Requires Python **3.11**

## Dev environment setup
```bash
# Make sure python and pip and venv installed
sudo apt install python3-venv python3-pip

# Setup virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

python3 --version 
> Python 3.11.10

# Install dependencies
pip install -r requirements.txt

# Run
python3 stream_processor.py
```

## Other commands / reference / notes
```bash

sudo apt install python3-venv python3-pip
source venv/bin/activate



pip install opencv-python-headless ultralytics torch torchvision



# Create a virtual environment, this creates a virtual environment in a folder named env.
python -m venv venv

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