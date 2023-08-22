#! /bin/bash

# Check for python
if ! hash python3; then
    echo "python3 is not installed"
    exit 1
fi

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" -lt "36" ]; then
    echo "This script requires python3 to be >= 3.6.x"
    exit 1
fi

# Check that nohup is installed, install it if not
dpkg -s coreutils &> /dev/null
if [ $? -ne 0 ]; then
	echo "nohup is not installed but is required, the script will install it"
	sudo apt install coreutils
fi

# Clean old environment
echo "Cleaning old environment"
rm -rf ~/poseidon_dec/
rm -rf nohup.out
echo "Killing old notebook"
pkill -f "jupyter.*8890"

# Create and activate new environment
echo "Creating fresh virtual environment"
python3 -m venv ~/poseidon_dec
source ~/poseidon_dec/bin/activate

# Install required packages
echo "Installing required packages"

pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade jupyterlab
pip install --upgrade ipywidgets
pip install --upgrade numpy talos pandas matplotlib
pip install --upgrade tensorflow_datasets
pip install --upgrade tensorflow_federated==0.18.0

# Launch jupyter
nohup jupyter notebook --no-browser --allow-root --port=8890 &
echo "Jupyter notebook launched on port 8890"

sleep 2

# Get notebook token
jupyter notebook list