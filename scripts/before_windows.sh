set -xe
echo "Running Windows build"
dir C:/ProgramData/chocolatey/bin/
dir C:/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/bin
export TF_USE_LOCAL_LIBRARY=1
curl -sS -o libpng.exe --insecure "https://downloads.sourceforge.net/project/gnuwin32/libpng/1.2.37/libpng-1.2.37-setup.exe?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fgnuwin32%2Ffiles%2Flibpng%2F1.2.37%2Flibpng-1.2.37-setup.exe%2Fdownload%3Fuse_mirror%3Dkent%26r%3Dhttp%253A%252F%252Fgnuwin32.sourceforge.net%252Fpackages%252Flibpng.htm%26use_mirror%3Dkent&ts=1557167610"
start libpng.exe
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
