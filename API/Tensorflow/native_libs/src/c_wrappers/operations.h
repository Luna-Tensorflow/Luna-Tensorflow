//
// Created by radeusgd on 01.12.18.
//

#ifndef TFL_OPERATIONS_H
#define TFL_OPERATIONS_H

#include <cstddef>
#include <tensorflow/c/c_api.h>

template <TF_DataType DataTypeLabel> class Tensor;
template <TF_DataType DataTypeLabel> class Operation;

extern "C" Operation<TF_FLOAT>* make_op_const_float(Tensor<TF_FLOAT>* tensor);
extern "C" Operation<TF_INT32>* make_op_const_int(Tensor<TF_INT32>* tensor);

extern "C" Operation<TF_FLOAT>* make_op_binary_float(const char* name, Operation<TF_FLOAT>* a, Operation<TF_FLOAT>* b);
extern "C" Operation<TF_INT32>* make_op_binary_int(const char* name, Operation<TF_INT32>* a, Operation<TF_INT32>* b);

extern "C" size_t operation_hashcode_float(Operation<TF_FLOAT>*);
extern "C" size_t operation_hashcode_int(Operation<TF_INT32>*);

extern "C" Tensor<TF_FLOAT>* eval_op_float(Operation<TF_FLOAT>* op);
extern "C" Tensor<TF_INT32>* eval_op_int(Operation<TF_INT32>* op);

extern "C" Operation<TF_FLOAT>* make_op_partial_derivative(Tensor<TF_FLOAT>* a, Tensor<TF_FLOAT>* b);

#endif //FFITESTHELPER_OPERATIONS_H
