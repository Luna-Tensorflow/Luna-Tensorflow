#include <memory>

#include "operations.h"

#include <string>
#include <memory>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "../helpers/utils.h"
#include "../helpers/LifeTimeManager.h"

#include "../ops/Operation.h"
#include "../ops/Gradient.h"
#include "../ops/Attr.h"

#include "../state/Variable.h"
#include "API/Tensorflow/native_libs/src/state/Sequence.h"
#include "../state/State.h"

namespace {
	Output **make_op_helper(const char *name, std::vector<Output *> inputs, std::vector<std::shared_ptr<Attr>> attrs,
                            int noutputs, const char *chosen_name) {
		std::vector<std::shared_ptr<Output>> inputs_v(inputs.size());

		std::transform(inputs.begin(), inputs.end(), inputs_v.begin(), [](Output* input){ return LifetimeManager::instance().accessOwned(input); });

		std::vector<std::shared_ptr<Output>> outputs = Operation::make_operation(name, inputs_v, noutputs, attrs,
																				 chosen_name);

		auto **output_ptrs = static_cast<Output **>(std::malloc(sizeof(Output *) * outputs.size()));

		for (unsigned i = 0; i < outputs.size(); ++i) {
			output_ptrs[i] = LifetimeManager::instance().addOwnership(
					std::dynamic_pointer_cast<Output>(outputs[i]));
		}

		return output_ptrs;
	}

	template<typename T>
	T get_first_and_free(T* arr) {
		T ret = arr[0];
		free(arr);
		return ret;
	}
}

Output* make_variable(const char* name, Tensor* val)
{
	FFILOG(name, val);
	auto tensor_ptr = LifetimeManager::instance().accessOwned(val);
	std::string sname(name);

	auto out_var = Variable::make_variable(sname, tensor_ptr);

	auto r = LifetimeManager::instance().addOwnership(out_var);
	FFILOGANDRETURN(r, name, val);
}

State* make_empty_state(void)
{
	FFILOG("empty_state");
	auto state = State::make_empty();
	return LifetimeManager::instance().addOwnership(state);
}

Tensor* get_value_from_state(State* ptr, const char* name)
{
    FFILOG(ptr, name);
    auto stateptr = LifetimeManager::instance().accessOwned(ptr);
    auto tensor = ptr->get(std::string(name));
    if (tensor == nullptr) return nullptr;
    return LifetimeManager::instance().addOwnership(tensor);
}

Tensor** get_values_from_state(State* ptr, const char** names, size_t count)
{
    FFILOG(ptr, names, count);
    auto stateptr = LifetimeManager::instance().accessOwned(ptr);
	std::vector<std::string> names_v(count);
	std::transform(names, names+count, names_v.begin(), [](auto name) {
	    return std::string(name);
	});

	auto returned = stateptr->get(names_v);

	return LifetimeManager::instance().addOwnershipOfArray(returned);
}

State* update_value_state(State* ptr, const char* name, const Tensor* newvalue)
{
    FFILOG(ptr, name, newvalue);
    std::string sname(name);
    auto tensorptr = LifetimeManager::instance().accessOwned(newvalue);
    auto stateptr = LifetimeManager::instance().accessOwned(ptr);

    return LifetimeManager::instance().addOwnership(stateptr->updated(sname, tensorptr));
}

State* update_state(State* ptr, const char** names, const Tensor** newvalues, size_t count)
{
    FFILOG(ptr, names, newvalues, count);
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> valuation;
    valuation.reserve(count);

    for(size_t i=0; i<count; ++i)
        valuation.emplace_back(std::string(names[i]),
            LifetimeManager::instance().accessOwned(newvalues[i]));

    auto newvals = LifetimeManager::instance().accessOwned(ptr)->updated(valuation);
    return LifetimeManager::instance().addOwnership(newvals);
}

Output* make_sequence(Output* sideefect, Output* value)
{
	FFILOG(sideefect, value);
	auto sideefectPtr = LifetimeManager::instance().accessOwned(sideefect);
	auto valPtr = LifetimeManager::instance().accessOwned(value);

	auto newValPtr = Sequence::make_sequence(sideefectPtr, valPtr);

	return LifetimeManager::instance().addOwnership(newValPtr);
}

Output **make_op(const char *name, Output **inputs, int ninputs, int noutputs, std::vector<std::shared_ptr<Attr>>* attr_list, const char *chosen_name) {
	if (attr_list == nullptr) {
		return make_op_helper(name, std::vector<Output*>(inputs, inputs + ninputs), {}, noutputs, chosen_name);
	}
	return make_op_helper(name, std::vector<Output*>(inputs, inputs + ninputs), *LifetimeManager::instance().accessOwned(attr_list), noutputs, chosen_name);
}

Output *make_op_binary(const char *name, Output *a, Output *b) {
	FFILOG(name, a, b);
	return get_first_and_free(make_op_helper(name, {a, b}, {}, 1, ""));
}

Output *make_op_unary(const char *name, Output *a) {
	FFILOG(name, a);
	return get_first_and_free(make_op_helper(name, {a}, {}, 1, ""));
}

Output *make_op_partial_derivative(Output *a, Output *b) {
	FFILOG(a, b);
	std::shared_ptr<Output> a_cpp = LifetimeManager::instance().accessOwned(a);
	std::shared_ptr<Output> b_cpp = LifetimeManager::instance().accessOwned(b);
	auto out = Gradient::add_gradients({a_cpp}, {b_cpp}, {})[0];
	auto outBase = std::dynamic_pointer_cast<Output>(out);
	return LifetimeManager::instance().addOwnership(std::move(outBase));
}

Output *make_op_placeholder(const char* name, TF_DataType type) {
	FFILOG(name);
    return get_first_and_free(make_op_helper("Placeholder", {}, {std::make_shared<AttrType>("dtype", type)}, 1, name));
}

Output *make_op_const(Tensor *tensor) {
	FFILOG(tensor);
	auto tensor_owned = LifetimeManager::instance().accessOwned(tensor);
    return get_first_and_free(make_op_helper("Const", {}, {std::make_shared<AttrTensor>("value", *tensor_owned),
                                        std::make_shared<AttrType>("dtype", tensor_owned->getType())}, 1, ""));
}

size_t operation_hashcode(Output *op) {
	FFILOG(op);
	return op->hashcode();
}

Tensor *eval_op(Output *op) {
	FFILOG(op);
	return LifetimeManager::instance().addOwnership(op->eval());
}

Tensor** batch_eval_op(Output** outs, size_t count)
{
    FFILOG(outs, count);
	char suppress_tf_log[] = "TF_CPP_MIN_FFILOG_LEVEL=3";
	putenv(suppress_tf_log);

	GraphSession graph;

	for(size_t i=0; i<count; ++i)
		graph.add_fetched_output(graph.add_output(outs[i]));

	return LifetimeManager::instance().addOwnershipOfArray(graph.eval()->outputs);
}

Tensor** batch_eval_op_placeholders(Output** outs, size_t op_count,
		const char* ph_names[], Tensor** ph_values, size_t ph_count)
{
    FFILOG(outs, op_count, ph_names, ph_values, ph_count);
	char suppress_tf_log[] = "TF_CPP_MIN_LOG_LEVEL=3";
	putenv(suppress_tf_log);

	GraphSession graph;

	for(size_t i=0; i<op_count; ++i)
		graph.add_fetched_output(graph.add_output(outs[i]));

	std::map<std::string, std::shared_ptr<Tensor>> substitutions;
	for(size_t i=0; i<ph_count; ++i)
	{
		substitutions.emplace(std::string(ph_names[i]),
			LifetimeManager::instance().accessOwned(ph_values[i]));
	}

	auto r = graph.eval(substitutions, State::make_empty()); // TODO support for state
	return LifetimeManager::instance().addOwnershipOfArray(r->outputs);
}

GraphSession* make_graph_from_output(Output* out)
{
	FFILOG(out);
	auto graphPtr = std::make_shared<GraphSession>();
	graphPtr->add_fetched_output(graphPtr->add_output(out));

	return LifetimeManager::instance().addOwnership(graphPtr);
}

GraphSession* make_graph_from_outputs(Output** out, size_t output_count)
{
	FFILOG(out, output_count);
	auto graphPtr = std::make_shared<GraphSession>();
	for(size_t i=0; i<output_count; ++i)
		graphPtr->add_fetched_output(graphPtr->add_output(out[i]));

	return LifetimeManager::instance().addOwnership(graphPtr);
}

void** eval_graph(GraphSession *graph, State* state)
{
	FFILOG(graph, state);
	auto statptr = LifetimeManager::instance().accessOwned(state);
	auto result = graph->eval(statptr);

	auto retv = static_cast<void**>(malloc((1 + result->outputs.size()) * sizeof(void*)));

	retv[0] = static_cast<void*>(LifetimeManager::instance().addOwnership(result->result_state));
	std::transform(result->outputs.begin(), result->outputs.end(), retv + 1, [](auto tensorptr)
	{
		return static_cast<void*>(LifetimeManager::instance().addOwnership(tensorptr));
	});

	return retv;
}

void** eval_graph_with_placeholders(GraphSession *graph,
		const char **ph_names, Tensor **ph_values, size_t ph_count, State* state)
{
	FFILOG(graph, ph_names, ph_values, ph_count, state);
	auto statptr = LifetimeManager::instance().accessOwned(state);

	std::map<std::string, std::shared_ptr<Tensor>> substitutions;
	for(size_t i=0; i<ph_count; ++i)
	{
		substitutions.emplace(std::string(ph_names[i]),
							  LifetimeManager::instance().accessOwned(ph_values[i]));
	}

	auto result = graph->eval(substitutions, statptr);

	auto retv = static_cast<void**>(malloc((1 + result->outputs.size()) * sizeof(void*)));

	retv[0] = static_cast<void*>(LifetimeManager::instance().addOwnership(result->result_state));
	std::transform(result->outputs.begin(), result->outputs.end(), retv + 1, [](auto tensorptr)
	{
		return static_cast<void*>(LifetimeManager::instance().addOwnership(tensorptr));
	});

	return retv;
}

State* fold_eval(GraphSession* graph, const char** ph_names, size_t ph_count, Tensor** ph_values, State* initial,
                         size_t fold_count){
	FFILOG(graph, ph_names, ph_count, ph_values, initial, fold_count);

	std::map<std::string, std::shared_ptr<Tensor>> substitutions;
	std::shared_ptr<State> state = LifetimeManager::instance().accessOwned(initial);

	for(size_t epoch=0; epoch < fold_count; ++epoch)
	{
		for(size_t ph = 0; ph < ph_count; ++ ph)
			substitutions.emplace(std::string(ph_names[ph]),
			                      LifetimeManager::instance().accessOwned(ph_values[epoch * ph_count + ph]));
		state = graph->eval(substitutions, state)->result_state;
		substitutions.clear();
	}

	return LifetimeManager::instance().addOwnership(state);
}