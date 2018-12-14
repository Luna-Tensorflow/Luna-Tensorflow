#include "operations.h"

#include <string>
#include <memory>
#include <cstdio>
#include "../helpers/utils.h"
#include "../ops/Operation.h"
#include "../ops/Const.h"
#include "../ops/BinaryOperation.h"
#include "../ops/UnaryOperation.h"
#include "../ops/Gradient.h"
#include "../ops/Placeholder.h"
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
	 LOG(name, a, b);
    return make_op_binary(name, a, b);
}

Operation<TF_INT32>* make_op_binary_int(const char* name, Operation<TF_INT32>* a, Operation<TF_INT32>* b) {
	 LOG(name, a, b);
    return make_op_binary(name, a, b);
}

namespace {
    template<TF_DataType DataTypeLabel>
    Operation<DataTypeLabel> *make_op_unary(const char *name, Operation<DataTypeLabel> *a) {
        std::string name_cpp(name);
        std::shared_ptr<Operation<DataTypeLabel>> a_cpp = LifetimeManager::instance().accessOwned(a);
        auto op = std::make_shared<UnaryOperation<DataTypeLabel>>(name_cpp, a_cpp);
        auto opBase = std::dynamic_pointer_cast<Operation<DataTypeLabel>>(op);
        return LifetimeManager::instance().addOwnership(std::move(opBase));
    }
}

Operation<TF_FLOAT>* make_op_unary_float(const char* name, Operation<TF_FLOAT>* a) {
     LOG(name, a);
    return make_op_unary(name, a);
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
	LOG(a, b);
	return make_op_derivative(a, b);
}

namespace {
	template<TF_DataType DT>
	Operation<DT> *make_op_placeholder(const char* name) {
		auto placeholder = std::make_shared<Placeholder<DT>>(std::string(name));
		auto placeholderBase = std::dynamic_pointer_cast<Operation<DT>>(placeholder);
		return LifetimeManager::instance().addOwnership(std::move(placeholderBase));
	}
}

Operation<TF_FLOAT>* make_op_placeholder_float(const char* name) {
	LOG(name);
	return make_op_placeholder<TF_FLOAT>(name);
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
	 LOG(tensor);
    return make_op_const(tensor);
}

Operation<TF_INT32>* make_op_const_int(Tensor<TF_INT32>* tensor) {
	 LOG(tensor);
    return make_op_const(tensor);
}

namespace {
    template<TF_DataType DataTypeLabel>
    size_t operation_hashcode(Operation<DataTypeLabel> *op) {
        return op->hashcode();
    }
}

size_t operation_hashcode_float(Operation<TF_FLOAT>* op) {
	 LOG(op);
    return operation_hashcode(op);
}

size_t operation_hashcode_int(Operation<TF_INT32>* op) {
	 LOG(op);
    return operation_hashcode(op);
}

namespace {
    template<TF_DataType DataTypeLabel>
    Tensor<DataTypeLabel> *eval_op(Operation<DataTypeLabel> *op) {
        return LifetimeManager::instance().addOwnership(op->eval());
    }
}

Tensor<TF_FLOAT> *eval_op_float(Operation<TF_FLOAT> *op) {
	 LOG(op);
    return eval_op(op);
}

Tensor<TF_INT32> *eval_op_int(Operation<TF_INT32> *op) {
	 LOG(op);
    return eval_op(op);
}

namespace {
	template<TF_DataType DT>
	Tensor<DT>** batch_eval_op(Operation<DT>** ops, size_t count)
	{
		char suppress_tf_log[] = "TF_CPP_MIN_LOG_LEVEL=3";
		putenv(suppress_tf_log);

		GraphSession graph;

		for(size_t i=0; i<count; ++i)
			graph.add_output(graph.add_operation(ops[i]));

		return graph.eval<DT>();
	}

	template<TF_DataType DT>
	Tensor<DT>** batch_eval_op_placeholders(Operation<DT>** ops, size_t op_count,
		const char* ph_names[], Tensor<DT>** ph_values, size_t ph_count)
	{
		char suppress_tf_log[] = "TF_CPP_MIN_LOG_LEVEL=3";
		putenv(suppress_tf_log);

		GraphSession graph;

		for(size_t i=0; i<op_count; ++i)
			graph.add_output(graph.add_operation(ops[i]));

		std::map<std::string, std::shared_ptr<Tensor<DT>>> substitutions;
		for(size_t i=0; i<ph_count; ++i)
		{
			substitutions.emplace(std::string(ph_names[i]),
				LifetimeManager::instance().accessOwned(ph_values[i]));
		}

		return graph.eval<DT>(substitutions);
	}
}

Tensor<TF_FLOAT>** batch_eval_op_placeholders_float(Operation<TF_FLOAT>** ops, size_t count,
	const char* ph_names[], Tensor<TF_FLOAT>** ph_values, size_t ph_count)
{
	LOG(ops, count, ph_names, ph_values, ph_count);
	return batch_eval_op_placeholders(ops, count, ph_names, ph_values, ph_count);
}

Tensor<TF_FLOAT>** batch_eval_op_float(Operation<TF_FLOAT>** ops, size_t count)
{
	LOG(ops, count);
	return batch_eval_op(ops, count);
}

