#Getting the sources:
```
git clone -b APIv1 https://github.com/Luna-Tensorflow/Luna-Tensorflow.git
```

#Build the C++ library to be used:
```
cd API/Tensorflow/native_libs
mkdir build
cd build
cmake ../src
make
```

#Running Luna
On Windows, alternatively you can open FFiExample/native_libs/src in CLion and build it using built-in CMake.
Then you can launch Main.luna in the editor or from command line:
```
cd API/Tensorflow
luna run
```
