set -xe
echo "Running Windows build"
dir C:/ProgramData/chocolatey/bin/
dir C:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin
export TF_USE_LOCAL_LIBRARY=1
mkdir libpng
cd libpng
curl -sS -o libpng.7z --insecure https://netix.dl.sourceforge.net/project/libpng/libpng16/1.6.37/lpng1637.7z
7z x libpng.7z -y
dir
cd lpng1637
./configure --disable-dependency-tracking
make check
make install
cd ../..
mkdir tensorflow # download TF dependency, TODO downloading protobufs
cd tensorflow
curl -sS -o libtf.zip --insecure https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.12.0.zip
7z x libtf.zip -y
dir
cd ..
mkdir build # prepare folders
mkdir release
mkdir release/native_libs
mkdir release/native_libs/windows/
#cd scripts/ # download and patch the patching script
#mv Main.hs Main2.hs
#git clone https://github.com/luna/dataframes
#mv dataframes/scripts/* .
#mv Main2.hs Main.hs
#cd ..
cd build
echo "Will generate makefiles"
cmake -G "MinGW Makefiles" -DCMAKE_SH=CMAKE_SH-NOTFOUND  ../API/Tensorflow/native_libs/src/
cd ..
