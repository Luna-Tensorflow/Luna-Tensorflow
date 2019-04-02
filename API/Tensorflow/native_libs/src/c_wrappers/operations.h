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

TFL_API Tensor** batch_eval_op(Output** ops, size_t count, const char **outError);
TFL_API Tensor** batch_eval_op_placeholders(Output** ops, size_t count, const char** ph_names, Tensor** ph_values,
		size_t ph_count, const char **outError);

TFL_API GraphSession* make_graph_from_output(Output* output, const char **outError);
TFL_API GraphSession* make_graph_from_outputs(Output** outputs, size_t output_count, const char **outError);

TFL_API void** eval_graph(GraphSession* graph, State* state, const char **outError);
TFL_API void** eval_graph_with_placeholders(GraphSession* graph, const char** ph_names, Tensor** ph_values,
		size_t ph_count, State* state, const char **outError);

TFL_API State* fold_eval(GraphSession* graph, const char** ph_names, size_t ph_count, Tensor** ph_values, State* initial,
	size_t foldCount, const char **outError);

TFL_API Output* make_variable(const char* name, Tensor* val, const char **outError);
TFL_API Output* make_sequence(Output* sideefect, Output* value, const char **outError);

TFL_API State* make_empty_state(const char **outError);
TFL_API Tensor* get_value_from_state(State* ptr, const char* name, const char **outError);
TFL_API Tensor** get_values_from_state(State* ptr, const char** names, size_t count, const char **outError);
TFL_API Tensor** get_variable_values_from_state(State* ptr, const Output** vars, size_t count, const char **outError);

TFL_API State* update_value_state(State* ptr, const char* name, const Tensor* newvalue, const char **outError);
TFL_API State* update_state(State* ptr, const char** names, const Tensor** newvalues, size_t count, const char **outError);

TFL_API Output** make_op(const char *name, Output **inputs, int ninputs, int noutputs, std::vector<std::shared_ptr<Attr>> *attr_list, const char *chosen_name, const char **outError);
TFL_API Output* make_op_const(Tensor* tensor, const char **outError);
TFL_API Output* make_op_placeholder(const char* name, TF_DataType type, const char **outError);
TFL_API Output* make_op_binary(const char* name, Output* a, Output* b, const char **outError);
TFL_API Output* make_op_unary(const char* name, Output* a, const char **outError);
TFL_API Output* make_op_partial_derivative(Output* a, Output* b, const char **outError);
TFL_API size_t operation_hashcode(Output* op, const char **outError);
TFL_API Tensor* eval_op(Output* op, const char **outError);

TFL_API const char* get_operation_name(Output* output, const char** outError);

#ifdef __cplusplus
};
#endif

#endif //FFITESTHELPER_OPERATIONS_H
