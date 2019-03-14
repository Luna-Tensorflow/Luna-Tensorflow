echo "Deploying (Windows)"
set -x
cd build
cmake --build .
cd ../release/
cp -rv ../API/Tensorflow/src/ . # copy Luna srcs and metadata
cp -rv ../API/Tensorflow/.luna-package/ .
cp -rv ../API/Tensorflow/native_libs/windows native_libs/
cp ../tensorflow/lib/*.dll native_libs/windows/ -v
7z a -t7z tensorflow_luna_windows.7z * .luna-package
cd ..
mkdir artifacts
mv release/tensorflow_luna_windows.7z artifacts/
