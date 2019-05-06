set -xe
echo "Running Windows build"
curl -sS -o libpng.exe --insecure "https://netcologne.dl.sourceforge.net/project/gnuwin32/libpng/1.2.37/libpng-1.2.37-setup.exe"
mv libpng.exe C:/ProgramData/chocolatey/bin
dir C:/ProgramData/chocolatey/bin/
dir C:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin
export TF_USE_LOCAL_LIBRARY=1
libpng /silent
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
