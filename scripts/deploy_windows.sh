echo "Deploying (Windows)"
set -x
cd build
cmake --build .
cd ../release/
cp -r ../API/Tensorflow/src/ . # copy Luna srcs and metadata
cp -r ../API/Tensorflow/.luna-package/ .
cp -r ../API/Tensorflow/native_libs/windows native_libs/
cp ../tensorflow/*.dll native_libs/windows/
7z a -t7z tensorflow_luna_windows.7z * .luna-package
cd ..
mkdir artifacts
mv release/tensorflow_luna_windows.7z artifacts/
