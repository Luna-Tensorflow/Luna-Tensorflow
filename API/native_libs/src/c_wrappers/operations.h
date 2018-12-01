//
// Created by radeusgd on 01.12.18.
//

#ifndef FFITESTHELPER_OPERATIONS_H
#define FFITESTHELPER_OPERATIONS_H

#include <cstddef>

class Tensor;
class Operation;

extern "C" Operation* make_op_const(Tensor* tensor);
extern "C" Operation* make_op_binary(const char* name, Operation* a, Operation* b);

extern "C" size_t operation_hashcode(Operation*);


#endif //FFITESTHELPER_OPERATIONS_H
