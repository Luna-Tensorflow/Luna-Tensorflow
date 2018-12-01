First build the C++ library to be used:
```
cd FfiExample/native_libs
mkdir build
cd build
cmake ../src
make
```
On Windows, alternatively you can open FFiExample/native_libs/src in CLion and build it using built-in CMake.

Then you can launch Main.luna in the editor or from command line.
