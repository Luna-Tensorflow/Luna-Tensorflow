# Luna-Tensorflow

## Description

### Overview
Luna-Tensorflow is a simple reference project that uses Tensorflow C API to solve linear regression problem. Intention of writing it was to get familiar with Tensorflow library, because a good knowledge of TF C API will be necessary in creating a binding to Luna language.

### Features
Making arithmetic operations on tensors is achieved by adding nodes into computation graph. We achieved that using a wrapper on library function TF_NewOperation. Among others, implemented operation types are: variable, constant, assignment, placeholder, addition, substraction, multiplication, square and gradient.

## Installation

### Attention
Make sure that [Tensorflow C](https://www.tensorflow.org/install/lang_c) is downloaded and linked.

### To run this project:
```bash
git clone -b sample_project https://github.com/Luna-Tensorflow/Luna-Tensorflow.git
cd Luna-Tensorflow/src
cmake ..
make
```

## Usage

Initial values: A = -2, b = 3

```bash
./tf_example
Iteration 0: A = 5.0952, b = 2.8552

...


Iteration 24999: A = 12.0276, b = -32.9603
```

Reference values: A = 12, b = 33

## Background
Link to Tensorflow [tutorial](https://www.tensorflow.org/tutorials/keras/basic_regression) on which this reference project is based.

## Support and Contact
Discord [channel](https://discordapp.com/channels/401396655599124480/506819805031038984) for this Luna-Tensorflow project.

## Changelog and Project status
The C part of this project is completed for now, but later, it will be transferred into Luna.
