//
// Created by radeusgd on 01.12.18.
//

#ifndef TFL_OPERATIONS_H
#define TFL_OPERATIONS_H

#include <cstddef>
#include <tensorflow/c/c_api.h>
#include "common.h"

#include "../graph/GraphSession.h"

template <TF_DataType DataTypeLabel> class Tensor;
template <TF_DataType DataTypeLabel> class Operation;

#ifdef __cplusplus
extern "C"
{
#endif

// TODO remove old definitions after migration is complete
TFL_API Operation<TF_FLOAT>* make_op_const_float(Tensor<TF_FLOAT>* tensor);
TFL_API Operation<TF_INT32>* make_op_const_int(Tensor<TF_INT32>* tensor);

TFL_API Operation<TF_FLOAT>* make_op_placeholder_float(const char* name);

TFL_API Operation<TF_FLOAT>* make_op_binary_float(const char* name, Operation<TF_FLOAT>* a, Operation<TF_FLOAT>* b);
TFL_API Operation<TF_INT32>* make_op_binary_int(const char* name, Operation<TF_INT32>* a, Operation<TF_INT32>* b);

TFL_API Operation<TF_FLOAT>* make_op_unary_float(const char* name, Operation<TF_FLOAT>* a);

TFL_API size_t operation_hashcode_float(Operation<TF_FLOAT>*);
TFL_API size_t operation_hashcode_int(Operation<TF_INT32>*);

TFL_API Tensor<TF_FLOAT>* eval_op_float(Operation<TF_FLOAT>* op);
TFL_API Tensor<TF_INT32>* eval_op_int(Operation<TF_INT32>* op);

TFL_API Operation<TF_FLOAT>* make_op_partial_derivative_float(Operation<TF_FLOAT>* a, Operation<TF_FLOAT>* b);

TFL_API Tensor<TF_FLOAT>** batch_eval_op_float(Operation<TF_FLOAT>** ops, size_t count);
TFL_API Tensor<TF_FLOAT>** batch_eval_op_placeholders_float(Operation<TF_FLOAT>** ops, size_t count,
	const char** ph_names, Tensor<TF_FLOAT>** ph_values, size_t ph_count);

TFL_API GraphSession* make_graph_from_output_float(Operation<TF_FLOAT>* output);
TFL_API GraphSession* make_graph_from_outputs_float(Operation<TF_FLOAT>** output, size_t output_count);

TFL_API Tensor<TF_FLOAT>** eval_graph_float(GraphSession* graph);

TFL_API Tensor<TF_FLOAT>** eval_graph_with_placeholders_float(GraphSession* graph,
	const char** ph_names, Tensor<TF_FLOAT>** ph_values, size_t ph_count);

#define DEFINE_OPS(typelabel) \
TFL_API Operation<typelabel>* make_op_const_##typelabel(Tensor<typelabel>* tensor); \
TFL_API Operation<typelabel>* make_op_placeholder_##typelabel(const char* name); \
TFL_API Operation<typelabel>* make_op_binary_##typelabel(const char* name, Operation<typelabel>* a, Operation<typelabel>* b); \
TFL_API Operation<typelabel>* make_op_unary_##typelabel(const char* name, Operation<typelabel>* a); \
TFL_API Operation<typelabel>* make_op_partial_derivative_##typelabel(Operation<typelabel>* a, Operation<typelabel>* b); \
TFL_API size_t operation_hashcode_##typelabel(Operation<typelabel>*); \
TFL_API Tensor<typelabel>* eval_op_##typelabel(Operation<typelabel>* op);
// TODO eval_graph may temporarily use only one type, but it should eventually support evaluating outputs of multiple types
// this can be achieved with a complementary structure with add_output_TYPE, eval, get_output_value_TYPE and output ids
// it should be done in the next iteration

DEFINE_OPS(TF_FLOAT);
DEFINE_OPS(TF_DOUBLE);
DEFINE_OPS(TF_INT8);
DEFINE_OPS(TF_INT16);
DEFINE_OPS(TF_INT32);
DEFINE_OPS(TF_INT64);
DEFINE_OPS(TF_UINT8);
DEFINE_OPS(TF_UINT16);
DEFINE_OPS(TF_UINT32);
DEFINE_OPS(TF_UINT64);
DEFINE_OPS(TF_BOOL);
DEFINE_OPS(TF_STRING);
//DEFINE_OPS(TF_HALF);

#ifdef __cplusplus
};
#endif

#endif //FFITESTHELPER_OPERATIONS_H
