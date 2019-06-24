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
You should download a Tensorflow C API release from https://www.tensorflow.org/install/lang_c and install it in your system according to the instructions there.

```
cd API/Tensorflow/native_libs
mkdir build
cd build
cmake ../src
make
```

Alternatively you can open API/Tensorflow/native_libs/src in CLion and build it using built-in CMake.

## Running Luna
Then you can launch Main.luna in the Luna Studio or from command line:
```
cd API/Tensorflow
luna run
```

