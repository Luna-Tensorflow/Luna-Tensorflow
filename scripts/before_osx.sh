set -xe
echo "Running OSX build"
# sudo add-apt-repository -y "ppa:ubuntu-toolchain-r/test"
# sudo apt-get update -y
# sudo apt-get install -y patchelf gcc-7 g++-7
curl -sSL https://get.haskellstack.org/ | sudo sh
#export COMPILER=g++-7 # prepare compiler
#export CXX=${COMPILER}
#export CC=gcc-7
export TF_USE_LOCAL_LIBRARY=1
mkdir tensorflow # download TF dependency, TODO downloading protobufs
cd tensorflow
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.13.1.tar.gz -O libtensorflow.tar.gz
tar -xvf libtensorflow.tar.gz
chmod u+w lib/*
cd ..
mkdir build # prepare folders
mkdir release
mkdir release/native_libs
mkdir release/native_libs/macos/
cd scripts/ # download and patch the patching script
mv Main.hs Main2.hs
git clone https://github.com/luna/dataframes
cd dataframes
git checkout 5f9e34ab27a3a0eafbcd33182b40bda3e50ccb14 # TODO upgrade to new version of patcher
cd ..
mv dataframes/scripts/* .
mv Main2.hs Main.hs
cd ..
cd build
cmake ../API/Tensorflow/native_libs/src/
cd ..
