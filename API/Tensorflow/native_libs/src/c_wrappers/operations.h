//
// Created by radeusgd on 01.12.18.
//

#ifndef TFL_OPERATIONS_H
#define TFL_OPERATIONS_H

#include <cstddef>
#include <tensorflow/c/c_api.h>
#include "common.h"

#include "../graph/GraphSession.h"
#include "attributes.h"

class Tensor;
class Output;

#ifdef __cplusplus
extern "C"
{
#endif

TFL_API Tensor** batch_eval_op(Output** ops, size_t count);
TFL_API Tensor** batch_eval_op_placeholders(Output** ops, size_t count, const char** ph_names, Tensor** ph_values,
		size_t ph_count);

TFL_API GraphSession* make_graph_from_output(Output* output);
TFL_API GraphSession* make_graph_from_outputs(Output** outputs, size_t output_count);

TFL_API Tensor** eval_graph(GraphSession* graph);

TFL_API Tensor** eval_graph_with_placeholders(GraphSession* graph, const char** ph_names, Tensor** ph_values,
		size_t ph_count);

TFL_API void** make_variable(const char* name, Tensor* default_value);
TFL_API Output* make_assign(Output* unit, Variable* var, Output* value);

TFL_API Output** make_op(const char *name, Output **inputs, int ninputs, int noutputs, std::vector<std::shared_ptr<Attr>> *attr_list, const char *chosen_name);
TFL_API Output* make_op_const(Tensor* tensor);
TFL_API Output* make_op_placeholder(const char* name, TF_DataType type);
TFL_API Output* make_op_binary(const char* name, Output* a, Output* b);
TFL_API Output* make_op_unary(const char* name, Output* a);
TFL_API Output* make_op_partial_derivative(Output* a, Output* b);
TFL_API size_t operation_hashcode(Output* op);
TFL_API Tensor* eval_op(Output* op);

#ifdef __cplusplus
};
#endif

#endif //FFITESTHELPER_OPERATIONS_H
