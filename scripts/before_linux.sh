echo "Running Linux build"
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -y
sudo apt-get install -y patchelf gcc-7 g++-7
curl -sSL https://get.haskellstack.org/ | sudo sh
export COMPILER=g++-7 # prepare compiler
export CXX=${COMPILER}
export CC=gcc-7
export TF_USE_LOCAL_LIBRARY=1
"${CXX} --version"
mkdir tensorflow # download TF dependency, TODO downloading protobufs
cd tensorflow
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz -O libtensorflow.tar.gz
tar -xvf libtensorflow.tar.gz
chmod u+w lib/*
cd ..
mkdir build # prepare folders
mkdir release
mkdir release/native_libs
mkdir release/native_libs/linux/
cd scripts/ # download and patch the patching script
mv Main.hs Main2.hs
git clone https://github.com/luna/dataframes
mv dataframes/scripts/* .
mv Main2.hs Main.hs
cd ..
cd build
cmake ../API/Tensorflow/native_libs/src/
cd ..
