#include <memory>

#include "operations.h"

#include <string>
#include <memory>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "../helpers/utils.h"
#include "../helpers/LifeTimeManager.h"
#include "../helpers/error.h"

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

Output* make_variable(const char* name, Tensor* val, const char **outError)
{
	return TRANSLATE_EXCEPTION(outError) {
		FFILOG(name, val);
		auto tensor_ptr = LifetimeManager::instance().accessOwned(val);
		std::string sname(name);

		auto out_var = Variable::make_variable(sname, tensor_ptr);

		auto r = LifetimeManager::instance().addOwnership(out_var);
		FFILOGANDRETURN(r, name, val);
	};
}

State* make_empty_state(const char **outError)
{
	return TRANSLATE_EXCEPTION(outError) {
		FFILOG("empty_state");
		auto state = State::make_empty();
		return LifetimeManager::instance().addOwnership(state);
	};
}

Tensor* get_value_from_state(State* ptr, const char* name, const char **outError)
{
	return TRANSLATE_EXCEPTION(outError) {
		FFILOG(ptr, name);
		auto stateptr = LifetimeManager::instance().accessOwned(ptr);
        std::string namestr = name;
		auto tensor = ptr->get(namestr);
		if (tensor == nullptr) throw std::runtime_error("No value found in state for " + namestr);
		return LifetimeManager::instance().addOwnership(tensor);
	};
}

Tensor** get_values_from_state(State* ptr, const char** names, size_t count, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(ptr, names, count);
        auto stateptr = LifetimeManager::instance().accessOwned(ptr);
        std::vector<std::string> names_v(count);
        std::transform(names, names+count, names_v.begin(), [](auto name) {
            return std::string(name);
        });

        auto returned = stateptr->get(names_v);
        for (size_t i = 0; i < returned.size(); ++i) {
            if (returned[i] == nullptr) {
                throw std::runtime_error("No value found in state for " + names_v[i]);
            }
        }

        return LifetimeManager::instance().addOwnershipOfArray(returned);
    };
}

Tensor** get_variable_values_from_state(State* ptr, const Output** vars, size_t count, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(ptr, vars, count);
        auto stateptr = LifetimeManager::instance().accessOwned(ptr);
        auto outsarray = LifetimeManager::instance().accessOwnedArray(vars, count);

        auto returned = stateptr->get_with_defaults(outsarray);

        return LifetimeManager::instance().addOwnershipOfArray(returned);
    };
}

State* update_value_state(State* ptr, const char* name, const Tensor* newvalue, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(ptr, name, newvalue);
        std::string sname(name);
        auto tensorptr = LifetimeManager::instance().accessOwned(newvalue);
        auto stateptr = LifetimeManager::instance().accessOwned(ptr);

        return LifetimeManager::instance().addOwnership(stateptr->updated(sname, tensorptr));
    };
}

State* update_state(State* ptr, const char** names, const Tensor** newvalues, size_t count, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(ptr, names, newvalues, count);
        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> valuation;
        valuation.reserve(count);

        for(size_t i=0; i<count; ++i)
            valuation.emplace_back(std::string(names[i]),
                LifetimeManager::instance().accessOwned(newvalues[i]));

        auto newvals = LifetimeManager::instance().accessOwned(ptr)->updated(valuation);
        return LifetimeManager::instance().addOwnership(newvals);
    };
}

Output* make_sequence(Output* sideefect, Output* value, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(sideefect, value);
        auto sideefectPtr = LifetimeManager::instance().accessOwned(sideefect);
        auto valPtr = LifetimeManager::instance().accessOwned(value);

        auto newValPtr = Sequence::make_sequence(sideefectPtr, valPtr);

        return LifetimeManager::instance().addOwnership(newValPtr);
    };
}

Output **make_op(const char *name, Output **inputs, int ninputs, int noutputs, std::vector<std::shared_ptr<Attr>>* attr_list, const char *chosen_name, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(name, inputs, ninputs, noutputs, attr_list, chosen_name);
        if (attr_list == nullptr) {
            return make_op_helper(name, std::vector<Output *>(inputs, inputs + ninputs), {}, noutputs, chosen_name);
        }
        return make_op_helper(name, std::vector<Output *>(inputs, inputs + ninputs),
                              *LifetimeManager::instance().accessOwned(attr_list), noutputs, chosen_name);
    };
}

Output *make_op_binary(const char *name, Output *a, Output *b, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(name, a, b);
        return get_first_and_free(make_op_helper(name, {a, b}, {}, 1, ""));
    };
}

Output *make_op_unary(const char *name, Output *a, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(name, a);
        return get_first_and_free(make_op_helper(name, {a}, {}, 1, ""));
    };
}

Output *make_op_partial_derivative(Output *a, Output *b, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(a, b);
        std::shared_ptr<Output> a_cpp = LifetimeManager::instance().accessOwned(a);
        std::shared_ptr<Output> b_cpp = LifetimeManager::instance().accessOwned(b);
        auto out = Gradient::add_gradients({a_cpp}, {b_cpp}, {})[0];
        auto outBase = std::dynamic_pointer_cast<Output>(out);
        return LifetimeManager::instance().addOwnership(std::move(outBase));
    };
}

Output *make_op_placeholder(const char* name, TF_DataType type, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(name);
        return get_first_and_free(make_op_helper("Placeholder", {}, {std::make_shared<AttrType>("dtype", type)}, 1, name));
    };
}

Output *make_op_const(const char* name, Tensor *tensor, const char **outError) {
	return TRANSLATE_EXCEPTION(outError) {
		FFILOG(name, tensor);
		auto tensor_owned = LifetimeManager::instance().accessOwned(tensor);
		return get_first_and_free(make_op_helper("Const", {}, {std::make_shared<AttrTensor>("value", *tensor_owned),
											std::make_shared<AttrType>("dtype", tensor_owned->getType())}, 1, name));
	};
}

namespace {
    template<TF_DataType TypeLabel> Tensor make_tensor_from_real_helper(double value) {
        auto x = static_cast<typename Type<TypeLabel>::tftype>(value);
        return Tensor(&x, nullptr, 0, TypeLabel);
    }

    Tensor make_tensor_from_real(double value, TF_DataType type) {
        switch (type) {
            case TF_FLOAT: return make_tensor_from_real_helper<TF_FLOAT>(value);
            case TF_DOUBLE: return make_tensor_from_real_helper<TF_DOUBLE>(value);
            case TF_INT32: return make_tensor_from_real_helper<TF_INT32>(value);
            case TF_INT64: return make_tensor_from_real_helper<TF_INT64>(value);
            case TF_UINT32: return make_tensor_from_real_helper<TF_UINT32>(value);
            case TF_UINT64: return make_tensor_from_real_helper<TF_UINT64>(value);
            // TODO add more supported types
            default: throw std::runtime_error("This type is not supported by constFromReal");
        }
    }
}

Output* make_op_const_from_real(const char* name, TF_DataType type, double value, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(name, type, value);
        auto tensor = make_tensor_from_real(value, type);
        return get_first_and_free(make_op_helper("Const", {}, {std::make_shared<AttrTensor>("value", tensor),
                                                               std::make_shared<AttrType>("dtype", tensor.getType())}, 1, name));
    };
}

size_t operation_hashcode(Output *op, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(op);
        auto ptr = LifetimeManager::instance().accessOwned(op);
        return ptr->hashcode();
    };
}

Tensor *eval_op(Output *op, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(op);
        auto ptr = LifetimeManager::instance().accessOwned(op);
        return LifetimeManager::instance().addOwnership(ptr->eval());
    };
}

Tensor** batch_eval_op(Output** outs, size_t count, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(outs, count);
        char suppress_tf_log[] = "TF_CPP_MIN_FFILOG_LEVEL=3";
        putenv(suppress_tf_log);

        GraphSession graph;

        for (size_t i = 0; i < count; ++i)
            graph.add_fetched_output(graph.add_output(outs[i]));

        return LifetimeManager::instance().addOwnershipOfArray(graph.eval()->outputs);
    };
}

Tensor** batch_eval_op_placeholders(Output** outs, size_t op_count,
		const char* ph_names[], Tensor** ph_values, size_t ph_count, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
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

        auto r = graph.eval(substitutions, State::make_empty());
        return LifetimeManager::instance().addOwnershipOfArray(r->outputs);
    };
}

GraphSession* make_graph_from_output(Output* out, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(out);
        auto graphPtr = std::make_shared<GraphSession>();
        graphPtr->add_fetched_output(graphPtr->add_output(out));

        return LifetimeManager::instance().addOwnership(graphPtr);
    };
}

GraphSession* make_graph_from_outputs(Output** out, size_t output_count, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(out, output_count);
        auto graphPtr = std::make_shared<GraphSession>();
        for(size_t i=0; i<output_count; ++i)
            graphPtr->add_fetched_output(graphPtr->add_output(out[i]));

        return LifetimeManager::instance().addOwnership(graphPtr);
    };
}

void** eval_graph(GraphSession *graph, State* state, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
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
    };
}

void** eval_graph_with_placeholders(GraphSession *graph,
		const char **ph_names, Tensor **ph_values, size_t ph_count, State* state, const char **outError)
{
    return TRANSLATE_EXCEPTION(outError) {
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
    };
}

#define CASE(typelabel) case typelabel: return static_cast<double>(tensor->at<typelabel>(index));

namespace {
    double to_double(std::shared_ptr<Tensor> tensor, int64_t index) {
        switch(tensor->getType()) {
            CASE(TF_FLOAT)
            CASE(TF_DOUBLE)
            CASE(TF_INT8)
            CASE(TF_INT16)
            CASE(TF_INT32)
            CASE(TF_INT64)
            CASE(TF_UINT8)
            CASE(TF_UINT16)
            CASE(TF_UINT32)
            CASE(TF_UINT64)
            default:
                throw std::runtime_error("Loss type not supported");
        }
    }

    double iterate_over_inputs(std::shared_ptr<GraphSession> &graph, const char** ph_names, Tensor** ph_values, uint32_t ph_count,
            uint32_t start_index, uint32_t end_index, bool apply_side_effects = true) {
        std::map<std::string, std::shared_ptr<Tensor>> substitutions;
        double running_loss = 0;
        for (uint32_t input_index = start_index; input_index < end_index; ++input_index) {
            for (size_t ph = 0; ph < ph_count; ++ph) {
                substitutions.emplace(std::string(ph_names[ph]),
                                      LifetimeManager::instance().accessOwned(ph_values[input_index * ph_count + ph]));
            }
            std::vector<std::shared_ptr<Tensor>> eval_results = graph->eval_one_step(substitutions, apply_side_effects);
            running_loss += to_double(eval_results[0], 0);
            substitutions.clear();
        }
        return running_loss / (end_index - start_index);
    }
}

void** train(GraphSession* graphPtr, const char** ph_names, size_t ph_count, Tensor** ph_values, State* initial,
                 size_t inputs_count, uint32_t epochs, uint32_t validation_samples, uint32_t early_stop,
                 const char **outError){
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(graph, ph_names, ph_count, ph_values, initial, inputs_count, epochs, validation_samples, early_stop);

        std::map<std::string, std::shared_ptr<Tensor>> substitutions;
        std::shared_ptr<State> state = LifetimeManager::instance().accessOwned(initial);
        std::vector<double> loss_history;
        std::shared_ptr<GraphSession> graph = LifetimeManager::instance().accessOwned(graphPtr);

        graph->initialize_variables(state);

        void **ret = static_cast<void**>(malloc(3 * sizeof(void*))); // needs to be freed by caller
        auto actual_epochs = static_cast<uint32_t*>(malloc(sizeof(uint32_t))); // needs to be freed by caller
        ret[2] = actual_epochs;

        uint32_t epochs_without_improvement = 0;

        for (*actual_epochs = 0; *actual_epochs < epochs && (epochs_without_improvement < early_stop || early_stop == 0); ++*actual_epochs) {
            double training_loss = iterate_over_inputs(graph, ph_names, ph_values, ph_count, validation_samples,
                    inputs_count, true);
            if (validation_samples == 0) {
                loss_history.push_back(training_loss);
            } else {
                loss_history.push_back(iterate_over_inputs(graph, ph_names, ph_values, ph_count, 0,
                        validation_samples, false));
            }
            if (early_stop > 0 && *actual_epochs > 0) {
                if (loss_history[*actual_epochs] < loss_history[*actual_epochs - 1]) {
                    epochs_without_improvement = 0;
                } else {
                    ++epochs_without_improvement;
                }
            }
        }

        ret[1] = malloc(*actual_epochs * sizeof(double)); // needs to be freed by caller
        memcpy(ret[1], loss_history.data(), *actual_epochs * sizeof(double));

        state = state->updated(graph->read_variables());

        ret[0] = static_cast<void*>(LifetimeManager::instance().addOwnership(state));

        return ret;
    };
}

State* fold_eval(GraphSession* graphPtr, const char** ph_names, size_t ph_count, Tensor** ph_values, State* initial,
                         size_t inputs_count, uint32_t epochs, const char **outError){
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(graph, ph_names, ph_count, ph_values, initial, inputs_count, epochs);

        std::map<std::string, std::shared_ptr<Tensor>> substitutions;
        std::shared_ptr<State> state = LifetimeManager::instance().accessOwned(initial);
        std::shared_ptr<GraphSession> graph = LifetimeManager::instance().accessOwned(graphPtr);

        graph->initialize_variables(state);

        for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t input_index = 0; input_index < inputs_count; ++input_index) {
                for (size_t ph = 0; ph < ph_count; ++ph) {
                    substitutions.emplace(std::string(ph_names[ph]),
                                          LifetimeManager::instance().accessOwned(ph_values[input_index * ph_count + ph]));
                }
                graph->eval_one_step(substitutions);
                substitutions.clear();
            }
        }

        state = state->updated(graph->read_variables());

        return LifetimeManager::instance().addOwnership(state);
    };
}

const char* get_operation_name(Output* output, const char** outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(output);
        auto o = LifetimeManager::instance().accessOwned(output);
        const char* name = o->get_binder()->get_name();
        if (name == nullptr) {
            throw std::runtime_error("This operation doesn't support `get_name`");
        }
        return name;
    };
}
