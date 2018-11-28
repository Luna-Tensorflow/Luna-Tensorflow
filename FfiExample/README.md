A minimal example of using Luna FFI. It might not work on Windows right now. First build the C++ library to be used:
```
cd FfiExample/native_libs
mkdir build
cd build
cmake ../src
make
```
Then you can launch Main.luna in the editor or from command line.