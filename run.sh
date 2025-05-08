# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then

    # Check if Python has been installed
    if ! command -v python3 &> /dev/null; then
        echo "Python is not found. Please install Python and ensure it's added to the PATH."
        exit 1
    fi

    python3 -m venv venv

fi

# Activate the virtual environment
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export PYSPARK_PYTHON=venv/bin/python
export PYSPARK_DRIVER_PYTHON=venv/bin/python

streamlit run app.py
