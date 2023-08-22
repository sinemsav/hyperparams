#! /bin/bash

echo "Killing notebooks"
pkill -f "jupyter.*8887"

echo "Remove environments and files"
rm -rf poseidon_iid/
rm -rf __pycache__/
rm -rf nohup.out