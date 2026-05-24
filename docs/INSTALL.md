# Installing NVIDIA Drivers and CUDA for RTX 3060

## Step 1: Check Hardware
```bash
lspci | grep -i nvidia
```
Should show your RTX 3060.

## Step 2: Install NVIDIA Driver (Ubuntu/Debian example)
```bash
# Add graphics-drivers PPA (optional for newer drivers)
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install driver (e.g., 550 series for RTX 3060)
sudo apt install nvidia-driver-550
sudo reboot
```

Verify:
```bash
nvidia-smi
```

## Step 3: Install CUDA Toolkit
Download from https://developer.nvidia.com/cuda-downloads (choose Linux -> x86_64 -> Ubuntu -> your version -> deb (local))

Example for Ubuntu 22.04:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

Add to PATH and LD_LIBRARY_PATH (usually done automatically by the deb installer, but ensure):
```bash
echo 'export PATH=/usr/local/cuda-12/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
nvcc --version
```

## Step 4: Install cuDNN (optional for deep learning)
Download from NVIDIA developer site (requires login) matching your CUDA version, then:
```bash
sudo dpkg -i libcudnn8_*.deb
sudo dpkg -i libcudnn8-dev_*.deb
```

## Step 5: Verify Installation
```bash
nvidia-smi
nvcc --version
```
You should see driver version, CUDA version, and your GPU listed.

## Step 6: Install Python Packages
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate tensorrt
```
(Adjust cu version to match your installed CUDA.)
