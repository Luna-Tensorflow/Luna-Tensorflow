echo "Deploying (Linux)"
set -xe
cd build
cmake --build .
cd ..
cd scripts/ # patch binaries
stack run -- ../release/native_libs/macos/ ../API/Tensorflow/native_libs/TODO
cd ../release/
cp -r ../API/Tensorflow/src/ . # copy Luna srcs and metadata
cp -r ../API/Tensorflow/.luna-package/ .
tar -cvf tensorflow_luna_osx.tar.gz .luna-package * # package it all
cd ..
mkdir artifacts
mv release/tensorflow_luna_osx.tar.gz artifacts/
