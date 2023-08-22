#! /bin/bash

# Check for python
if ! hash python3; then
    echo "python3 is not installed"
    exit 1
fi

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" -lt "36" && "$ver" -gt "37" ]; then
    echo "This script requires python3 to be >= 3.6.x and <= 3.7.x"
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
rm -rf poseidon_iid/
rm -rf nohup.out
echo "Killing old notebook"
pkill -f "jupyter.*8887"

# Create and activate new environment
echo "Creating fresh virtual environment"
python3 -m venv poseidon_iid
sleep 2
source poseidon_iid/bin/activate

# Install required packages
echo "Installing required packages"

pip install --upgrade pip &> /dev/null
echo -ne "1/9 \r"
pip install --upgrade wheel &> /dev/null
echo -ne "2/9 \r"
pip install --upgrade jupyterlab &> /dev/null
echo -ne "3/9 \r"
pip install --upgrade ipywidgets &> /dev/null
echo -ne "4/9 \r"
pip install --upgrade tensorflow==2.4.1 &> /dev/null
echo -ne "5/9 \r"
pip install --upgrade numpy talos pandas &> /dev/null
echo -ne "8/9 \r"
pip install --upgrade tensorflow_datasets &> /dev/null
echo -ne "9/9 \r"
echo -ne "\n"

# Replace files
./iid_replace.sh

# Launch jupyter
nohup jupyter notebook --no-browser --allow-root --port=8887 &
echo "Jupyter notebook launched on port 8887"

sleep 2

# Get notebook token
jupyter notebook list