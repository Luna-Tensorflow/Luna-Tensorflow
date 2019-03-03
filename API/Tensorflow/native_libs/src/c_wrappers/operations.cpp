#include <memory>

#include "operations.h"

#include <string>
#include <memory>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "../helpers/utils.h"
#include "../ops/Operation.h"
#include "../helpers/LifeTimeManager.h"
#include "../ops/Gradient.h"
#include "../ops/Attr.h"
#include "../state/Variable.h"
#include "../state/Assign.h"

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

void** make_variable(const char* name, Tensor* val)
{
	LOG(name, val);
	auto tensor_ptr = LifetimeManager::instance().accessOwned(val);
	std::string sname(name);

	auto arr = static_cast<void**> (malloc(2 * sizeof(void*)));
	auto out_var = Variable::make_variable(sname, tensor_ptr);

	arr[0] = static_cast<void*>(LifetimeManager::instance().addOwnership(out_var.first));
	arr[1] = static_cast<void*>(LifetimeManager::instance().addOwnership(out_var.second));

	return arr;
}

Output* make_assign(Output* unit, Variable* var, Output* value)
{
	LOG(unit, var, value);
	auto unitPtr = LifetimeManager::instance().accessOwned(unit);
	auto varPtr = LifetimeManager::instance().accessOwned(var);
	auto valPtr = LifetimeManager::instance().accessOwned(value);

	auto newUnitPtr = Assign::make_assign(unitPtr, varPtr, valPtr);

	return LifetimeManager::instance().addOwnership(newUnitPtr);
}

Output **make_op(const char *name, Output **inputs, int ninputs, int noutputs, std::vector<std::shared_ptr<Attr>>* attr_list, const char *chosen_name) {
	if (attr_list == nullptr) {
		return make_op_helper(name, std::vector<Output*>(inputs, inputs + ninputs), {}, noutputs, chosen_name);
	}
	return make_op_helper(name, std::vector<Output*>(inputs, inputs + ninputs), *LifetimeManager::instance().accessOwned(attr_list), noutputs, chosen_name);
}

Output *make_op_binary(const char *name, Output *a, Output *b) {
	LOG(name, a, b);
	return get_first_and_free(make_op_helper(name, {a, b}, {}, 1, ""));
}

Output *make_op_unary(const char *name, Output *a) {
	LOG(name, a);
	return get_first_and_free(make_op_helper(name, {a}, {}, 1, ""));
}

Output *make_op_partial_derivative(Output *a, Output *b) {
	LOG(a, b);
	std::shared_ptr<Output> a_cpp = LifetimeManager::instance().accessOwned(a);
	std::shared_ptr<Output> b_cpp = LifetimeManager::instance().accessOwned(b);
	auto out = Gradient::add_gradients({a_cpp}, {b_cpp}, {})[0];
	auto outBase = std::dynamic_pointer_cast<Output>(out);
	return LifetimeManager::instance().addOwnership(std::move(outBase));
}

Output *make_op_placeholder(const char* name, TF_DataType type) {
	LOG(name);
    return get_first_and_free(make_op_helper("Placeholder", {}, {std::make_shared<AttrType>("dtype", type)}, 1, name));
}

Output *make_op_const(Tensor *tensor) {
	LOG(tensor);
	auto tensor_owned = LifetimeManager::instance().accessOwned(tensor);
    return get_first_and_free(make_op_helper("Const", {}, {std::make_shared<AttrTensor>("value", *tensor_owned),
                                        std::make_shared<AttrType>("dtype", tensor_owned->getType())}, 1, ""));
}

size_t operation_hashcode(Output *op) {
	LOG(op);
	return op->hashcode();
}

Tensor *eval_op(Output *op) {
	LOG(op);
	return LifetimeManager::instance().addOwnership(op->eval());
}

Tensor** batch_eval_op(Output** outs, size_t count)
{
    LOG(outs, count);
	char suppress_tf_log[] = "TF_CPP_MIN_LOG_LEVEL=3";
	putenv(suppress_tf_log);

	GraphSession graph;

	for(size_t i=0; i<count; ++i)
		graph.add_fetched_output(graph.add_output(outs[i]));

	return LifetimeManager::instance().addOwnershipOfArray(graph.eval());
}

Tensor** batch_eval_op_placeholders(Output** outs, size_t op_count,
		const char* ph_names[], Tensor** ph_values, size_t ph_count)
{
    LOG(outs, op_count, ph_names, ph_values, ph_count);
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
	LOG(out);
	auto graphPtr = std::make_shared<GraphSession>();
	graphPtr->add_fetched_output(graphPtr->add_output(out));

	return LifetimeManager::instance().addOwnership(graphPtr);
}

GraphSession* make_graph_from_outputs(Output** out, size_t output_count)
{
	LOG(out, output_count);
	auto graphPtr = std::make_shared<GraphSession>();
	for(size_t i=0; i<output_count; ++i)
		graphPtr->add_fetched_output(graphPtr->add_output(out[i]));

	return LifetimeManager::instance().addOwnership(graphPtr);
}

Tensor** eval_graph(GraphSession *graph)
{
	LOG(graph);
	return LifetimeManager::instance().addOwnershipOfArray(graph->eval());
}

Tensor** eval_graph_with_placeholders(GraphSession *graph,
		const char **ph_names, Tensor **ph_values, size_t ph_count)
{
	LOG(graph, ph_names, ph_values, ph_count);
	std::map<std::string, std::shared_ptr<Tensor>> substitutions;
	for(size_t i=0; i<ph_count; ++i)
	{
		substitutions.emplace(std::string(ph_names[i]),
							  LifetimeManager::instance().accessOwned(ph_values[i]));
	}

	auto r = graph->eval(substitutions, State::make_empty()); // TODO support for state
	return LifetimeManager::instance().addOwnershipOfArray(r->outputs);
}