# Setup virtual environment.
sudo apt install python3-venv
python3 -m venv ./venv/
./venv/bin/python3 -m pip install --upgrade pip

# Some required packages for building.
sudo apt install -y cmake libeigen3-dev libsuitesparse-dev qt5-default python3-dev

# Install required packages.
./venv/bin/pip3 install -r requirements.txt

# Download and install g2opy (python bindings for g2o).
git clone https://github.com/uoip/g2opy.git
cd g2opy
mkdir build
cd build
cmake ..
make -j8
cd ..
cp lib/g2o.cpython-36m-x86_64-linux-gnu.so ../venv/lib/python3.6/site-packages/
