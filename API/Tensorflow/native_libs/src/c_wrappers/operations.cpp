#include "operations.h"

#include <string>
#include <memory>
#include <cstdio>
#include "../helpers/utils.h"
#include "../ops/Operation.h"
#include "../ops/Const.h"
#include "../ops/BinaryOperation.h"
#include "../helpers/LifeTimeManager.h"

namespace {
    template<TF_DataType DataTypeLabel>
    Operation<DataTypeLabel> *make_op_binary(const char *name, Operation<DataTypeLabel> *a, Operation<DataTypeLabel> *b) {
        std::string name_cpp(name);
        std::shared_ptr<Operation<DataTypeLabel>> a_cpp = LifetimeManager::instance().accessOwned(a);
        std::shared_ptr<Operation<DataTypeLabel>> b_cpp = LifetimeManager::instance().accessOwned(b);
        auto op = std::make_shared<BinaryOperation<DataTypeLabel>>(name_cpp, a_cpp, b_cpp);
        auto opBase = std::dynamic_pointer_cast<Operation<DataTypeLabel>>(op);
        return LifetimeManager::instance().addOwnership(std::move(opBase));
    }
}

Operation<TF_FLOAT>* make_op_binary_float(const char* name, Operation<TF_FLOAT>* a, Operation<TF_FLOAT>* b) {
    return make_op_binary(name, a, b);
}

Operation<TF_INT32>* make_op_binary_int(const char* name, Operation<TF_INT32>* a, Operation<TF_INT32>* b) {
    return make_op_binary(name, a, b);
}


namespace {
	template<TF_DataType DT>
	Operation<DT> *make_op_derivative(Operation<DT> *a, Operation<DT> *b) {
		std::shared_ptr<Operation<DT>> a_cpp = LifetimeManager::instance().accessOwned(a);
		std::shared_ptr<Operation<DT>> b_cpp = LifetimeManager::instance().accessOwned(b);
		auto op = std::make_shared<Gradient<DT>>(a_cpp, b_cpp);
		auto opBase = std::dynamic_pointer_cast<Operation<DT>>(op);
		return LifetimeManager::instance().addOwnership(std::move(opBase));
	}
}

Operation<TF_FLOAT>* make_op_partial_derivative(Operation<TF_FLOAT>* a, Operation<TF_FLOAT> *b) {
	return make_op_derivative(a, b);
}

namespace {
    template<TF_DataType DT>
    Operation<DT> *make_op_const(Tensor<DT> *tensor) {
        std::shared_ptr<Tensor<DT>> tensor_cpp = LifetimeManager::instance().accessOwned(tensor);
        auto constant = std::make_shared<Const<DT>>(tensor_cpp);
        auto constantBase = std::dynamic_pointer_cast<Operation<DT>>(constant);
        return LifetimeManager::instance().addOwnership(std::move(constantBase));
    }
}

Operation<TF_FLOAT>* make_op_const_float(Tensor<TF_FLOAT>* tensor) {
    return make_op_const(tensor);
}

Operation<TF_INT32>* make_op_const_int(Tensor<TF_INT32>* tensor) {
    return make_op_const(tensor);
}

namespace {
    template<TF_DataType DataTypeLabel>
    size_t operation_hashcode(Operation<DataTypeLabel> *op) {
        return op->hashcode();
    }
}

size_t operation_hashcode_float(Operation<TF_FLOAT>* op) {
    return operation_hashcode(op);
}

size_t operation_hashcode_int(Operation<TF_INT32>* op) {
    return operation_hashcode(op);
}

namespace {
    template<TF_DataType DataTypeLabel>
    Tensor<DataTypeLabel> *eval_op(Operation<DataTypeLabel> *op) {
        return LifetimeManager::instance().addOwnership(op->eval());
    }
}

Tensor<TF_FLOAT> *eval_op_float(Operation<TF_FLOAT> *op) {
    return eval_op(op);
}

Tensor<TF_INT32> *eval_op_int(Operation<TF_INT32> *op) {
    return eval_op(op);
}

namespace {
	template<TF_DataType DT>
	Tensor<DT>** batch_eval_op(Operation<DT>** ops, size_t count)
	{
		char suppress_tf_log[] = "TF_CPP_MIN_LOG_LEVEL=3";
		putenv(suppress_tf_log);

		GraphSession graph;

		std::vector<TF_Tensor*> output_values(count);
		std::vector<TF_Output> tf_outputs(count);

		for(unsigned i=0; i<count; ++i)
			tf_outputs[i] = graph.add(ops[i]);

		run_with_status<void>(std::bind(TF_SessionRun,
		                                graph.get_underlying_session(),
		                                nullptr,
		                                nullptr, nullptr, 0,
		                                tf_outputs.data(), output_values.data(), count,
		                                nullptr, 0,
		                                nullptr,
		                                std::placeholders::_1));

		auto return_values = (Tensor<DT>**) std::calloc(sizeof(Tensor<DT>*), count);

		for(unsigned i=0; i<count; ++i)
		{
			return_values[i] = LifetimeManager::instance().addOwnership(std::make_shared<Tensor<DT>>(output_values[i]));
		}

		return return_values;
	}
}

Tensor<TF_FLOAT>** batch_eval_op_float(Operation<TF_FLOAT>** ops, size_t count)
{
	return batch_eval_op(ops, count);
}