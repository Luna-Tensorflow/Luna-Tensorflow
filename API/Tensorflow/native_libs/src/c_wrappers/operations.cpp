#include "operations.h"

#include <string>
#include <memory>
#include "../utils.h"
#include "../ops/Operation.h"
#include "../ops/Const.h"
#include "../ops/BinaryOperation.h"
#include "../LifeTimeManager.h"

Operation* make_op_binary(const char* name, Operation* a, Operation* b) {
    std::string name_cpp(name);
    std::shared_ptr<Operation> a_cpp = LifetimeManager::instance().accessOwned(a);
    std::shared_ptr<Operation> b_cpp = LifetimeManager::instance().accessOwned(b);
    auto op = std::make_shared<BinaryOperation>(name_cpp, a_cpp, b_cpp);
    auto opBase = std::dynamic_pointer_cast<Operation>(op);
    return LifetimeManager::instance().addOwnership(std::move(opBase));
}

template<TF_DataType DT> Operation* make_op_const(Tensor<DT>* tensor) {
    std::shared_ptr<Tensor<DT>> tensor_cpp = LifetimeManager::instance().accessOwned(tensor);
    auto constant = std::make_shared<Const<DT>>(tensor_cpp);
    auto constantBase = std::dynamic_pointer_cast<Operation>(constant);
    return LifetimeManager::instance().addOwnership(std::move(constantBase));
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