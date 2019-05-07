echo "Deploying (Windows)"
set -xe
cd build
cmake --build .
cd ../release/
cp -rv ../API/Tensorflow/src/ . # copy Luna srcs and metadata
cp -rv ../API/Tensorflow/.luna-package/ .
cp -v ../API/Tensorflow/native_libs/windows/libTFL.dll native_libs/windows/
cp -v ../tensorflow/lib/tensorflow.dll native_libs/windows/libtensorflow.so # weird naming but that's what libTFL.dll requires
cp -v ../zlib/zlib-1.2.11/libzlib.dll native_libs/windows/
cp -v ../libpng/lpng1637/libpng.dll native_libs/windows/
cp -v C:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin/libgcc_s_seh-1.dll native_libs/windows/
cp -v C:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin/libwinpthread-1.dll native_libs/windows/
cp -v C:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin/libstdc++-6.dll native_libs/windows/
7z a -t7z tensorflow_luna_windows.7z * .luna-package
cd ..
mkdir artifacts
mv release/tensorflow_luna_windows.7z artifacts/
