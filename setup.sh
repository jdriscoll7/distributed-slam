# Setup virtual environment.
python3 -m venv ./venv/
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

