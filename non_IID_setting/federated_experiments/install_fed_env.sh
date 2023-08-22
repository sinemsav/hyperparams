#! /bin/bash

if [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
    echo "cpu or gpu not specified"
    exit 1
fi

# Check for python
if ! hash python3; then
    echo "python3 is not installed"
    exit 1
fi

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" -lt "36" ]; then
    echo "This script requires python3 to be at least Python 3.6"
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
echo "Killing notebooks"
pkill -f "jupyter.*8890"
pkill -f "jupyter.*8891"

echo "Remove environments and files"
rm -rf ~/poseidon_fed/
rm -rf __pycache__/
rm -rf nohup.out

sleep 2

# Create and activate new environment
echo "Creating fresh virtual environment"
python3 -m venv ~/poseidon_fed
source ~/poseidon_fed/bin/activate

# Install required packages
echo "Installing required packages"

pip install --upgrade pip
pip install -U autopep8
pip install talos
pip install --upgrade wheel
pip install --upgrade jupyterlab
pip install --upgrade ipywidgets
pip install --upgrade numpy pandas matplotlib
pip install --upgrade tensorflow_datasets
pip install --upgrade tensorflow_federated

if [ "$1" == "cpu" ]; then
    pip install --upgrade "jax[cpu]"
elif [ "$1" == "gpu" ]; then
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
fi
pip install fedjax
echo -ne "\n"

# Launch jupyter
nohup jupyter notebook --no-browser --allow-root --port=8891 &
echo "Jupyter notebook launched on port 8891"

sleep 2

# Get notebook token
jupyter notebook list
