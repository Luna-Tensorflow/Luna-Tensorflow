echo "Deploying (Linux)"
set -xe
cd build
cmake --build .
cd ..
cd scripts/ # patch binaries
stack run -- ../release/native_libs/linux/ ../API/Tensorflow/native_libs/linux/libTFL.so
cd ../release/
cp -r ../API/Tensorflow/src/ . # copy Luna srcs and metadata
cp -r ../API/Tensorflow/.luna-package/ .
tar -cvf tensorflow_luna_linux.tar.gz .luna-package * # package it all
cd ..
mkdir artifacts
mv release/tensorflow_luna_linux.tar.gz artifacts/
