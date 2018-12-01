#include "operations.h"

#include <string>
#include <memory>
#include <API/native_libs/src/utils.h>
#include <API/native_libs/src/ops/Operation.h>
#include <API/native_libs/src/ops/Const.h>
#include <API/native_libs/src/LifeTimeManager.h>
#include <API/native_libs/src/ops/BinaryOperation.h>

Operation* make_op_binary(const char* name, Operation* a, Operation* b) {
    std::string name_cpp(name);
    std::shared_ptr<Operation> a_cpp = LifetimeManager::instance().accessOwned(a);
    std::shared_ptr<Operation> b_cpp = LifetimeManager::instance().accessOwned(b);
    std::shared_ptr<BinaryOperation> op = std::make_shared<BinaryOperation>(name_cpp, a_cpp, b_cpp);
    return LifetimeManager::instance().addOwnership(std::move(op));
}

template<TF_DataType DT> Operation* make_op_const(Tensor<DT>* tensor) {
    std::shared_ptr<Tensor<DT>> tensor_cpp = LifetimeManager::instance().accessOwned(tensor);
    std::shared_ptr<Const<DT>> constant = std::make_shared<Const<DT>>(tensor_cpp);
    return LifetimeManager::instance().addOwnership(std::move(constant));
}

Operation* make_op_const_float(Tensor<TF_FLOAT>* tensor) {
    return make_op_const(tensor);
}

Operation* make_op_const_int(Tensor<TF_INT32>* tensor) {
    return make_op_const(tensor);
}

size_t operation_hashcode(Operation* op) {
    return op->hashcode();
}