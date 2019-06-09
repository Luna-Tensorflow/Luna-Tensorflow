# Luna-Tensorflow

![](API/Tensorflow/logo.png)

## Description 

The following project is the binding of the <a href="https://www.tensorflow.org/">Tensorflow</a> library for the 
<a href="https://www.luna-lang.org/">Luna</a> programming language. 

## Getting the sources:
```
git clone https://github.com/Luna-Tensorflow/Luna-Tensorflow.git
```

## Building the internal C++ library:
```
cd API/Tensorflow/native_libs
mkdir build
cd build
cmake ../src
make
```

## Running Luna
On Windows, alternatively you can open FFiExample/native_libs/src in CLion and build it using built-in CMake.
Then you can launch Main.luna in the Luna Studio or from command line:
```
cd API/Tensorflow
luna run
```

