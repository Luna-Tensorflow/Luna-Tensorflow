//
// Created by radeusgd on 01.12.18.
//

#ifndef TFL_OPERATIONS_H
#define TFL_OPERATIONS_H

#include <cstddef>
#include <tensorflow/c/c_api.h>

template <TF_DataType DataTypeLabel> class Tensor;
class Operation;

extern "C" Operation* make_op_const_float(Tensor<TF_FLOAT>* tensor);
extern "C" Operation* make_op_const_int(Tensor<TF_INT32>* tensor);
extern "C" Operation* make_op_binary(const char* name, Operation* a, Operation* b);

extern "C" size_t operation_hashcode(Operation*);

#endif //FFITESTHELPER_OPERATIONS_H
