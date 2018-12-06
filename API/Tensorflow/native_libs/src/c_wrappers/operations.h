//
// Created by radeusgd on 01.12.18.
//

#ifndef TFL_OPERATIONS_H
#define TFL_OPERATIONS_H

#include <cstddef>
#include <tensorflow/c/c_api.h>

template <TF_DataType DataTypeLabel> class Tensor;
template <TF_DataType DataTypeLabel> class Operation;

#ifdef __cplusplus
extern "C"
{
#endif

Operation<TF_FLOAT>* make_op_const_float(Tensor<TF_FLOAT>* tensor);
Operation<TF_INT32>* make_op_const_int(Tensor<TF_INT32>* tensor);

Operation<TF_FLOAT>* make_op_binary_float(const char* name, Operation<TF_FLOAT>* a, Operation<TF_FLOAT>* b);
Operation<TF_INT32>* make_op_binary_int(const char* name, Operation<TF_INT32>* a, Operation<TF_INT32>* b);

size_t operation_hashcode_float(Operation<TF_FLOAT>*);
size_t operation_hashcode_int(Operation<TF_INT32>*);

Tensor<TF_FLOAT>* eval_op_float(Operation<TF_FLOAT>* op);
Tensor<TF_INT32>* eval_op_int(Operation<TF_INT32>* op);

Operation<TF_FLOAT>* make_op_partial_derivative(Operation<TF_FLOAT>* a, Operation<TF_FLOAT>* b);

Tensor<TF_FLOAT>** batch_eval_op_float(Operation<TF_FLOAT>** ops, size_t count);

#ifdef __cplusplus
};
#endif

#endif //FFITESTHELPER_OPERATIONS_H
