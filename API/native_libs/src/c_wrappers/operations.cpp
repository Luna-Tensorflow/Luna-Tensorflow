#include "operations.h"

#include <string>
#include <API/native_libs/src/utils.h>
#include <API/native_libs/src/ops/Operation.h>

Operation* make_op_binary(const char* name, Operation* a, Operation* b) {
    std::string cpp_name(name);
    return not_implemented<Operation*>();
}

Operation* make_op_const(Tensor* tensor) {
    return not_implemented<Operation*>();
}

size_t operation_hashcode(Operation* op) {
    return op->hashcode();
}