# IC Cluster Setup - UBUNTU 20.04

Run this command to avoid the cluster never reconnecting when it is rebooted:
`sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`

## Update system

```{bash}
sudo apt update
sudo apt upgrade
sudo apt -y install gnupg
```

## Install CUDA

To install CUDA, run:

```{bash}
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-5-local/7fa2af80.pub
sudo apt update
sudo apt -y install cuda

nano ~/.bashrc
export PATH="/usr/local/cuda-11.5/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.5/lib64:$LD_LIBRARY_PATH"
```

[Tutorial with additional details](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-nvidia-driver-and-cuda-software)

## Instal cuDNN

First, download this [file](https://drive.google.com/file/d/1gVovwfd58lmZS1VQSjNxWmkpxqJX8GMD/view?usp=sharing).

To install cuDNN, run:

```{bash}
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.3.1.22_1.0-1_amd64.deb
sudo apt-key add /var/cudnn-local-repo-*/7fa2af80.pub
sudo apt update
sudo apt -y install libcudnn8=8.3.1.22-1+cuda11.5

sudo reboot
```

## Set up repository

```{bash}
sudo apt -y install git
sudo apt -y install python3-venv

# Clone project repository
git clone https://github.com/ldsec/projects-data.git
```

## Run python scripts in parallel

The IC clusters contain 4 GPUs. However, the scripts only make use of one of them at a time. To run multiple python scripts in parallel, mask all GPUs except for one, with the command `export CUDA_VISIBLE_DEVICES=x`, where x is a number between 0 and 3. Afterwards, you can run another script making use of a different GPU while the previous Python process is still running.

## [Additional] Install Python with specific version

To install a specific Python version, run the following commands, where x is a number between 6 and 9.

```{bash}
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.x python3.x-dev python3.x-venv python-wheel
```

## [Additional] Uninstall CUDA and cuDNN

To completely remove CUDA and cuDNN, run the following commands:

```{bash}
sudo apt -y remove cuda
sudo apt -y remove libcudnn8=8.3.1.22-1+cuda11.5
sudo apt -y remove nvidia-driver-495
sudo apt purge nvidia-*
sudo apt autoremove
rm -r /usr/local/cuda-11.5/
rm -r /usr/local/cuda-11/
rm -r /usr/local/cuda/
```
