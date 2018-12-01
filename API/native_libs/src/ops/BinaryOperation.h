#ifndef FFITESTHELPER_BINARYOPERATION_H
#define FFITESTHELPER_BINARYOPERATION_H

#include <string>
#include <memory>
#include <utility>
#include <API/native_libs/src/utils.h>
#include "Operation.h"

class BinaryOperation : public Operation {
public:
    BinaryOperation(std::string name,
                    std::shared_ptr<Operation> a,
                    std::shared_ptr<Operation> b)
        : operation_name(std::move(name)), arg1(std::move(a)), arg2(std::move(b)) {
        hash = std::hash<std::string>()(operation_name);
        hash = hash_combine(hash, a->hashcode());
        hash = hash_combine(hash, b->hashcode());
    }

    size_t hashcode() override {
        return hash;
    }
private:
    size_t hash;
    std::string operation_name;
    std::shared_ptr<Operation> arg1, arg2;
};

extern "C" Operation* make_op_binary(const char* name, Operation* a, Operation* b);

#endif //FFITESTHELPER_BINARYOPERATION_H
