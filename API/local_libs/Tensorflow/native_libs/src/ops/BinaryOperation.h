#ifndef TFL_BINARYOPERATION_H
#define TFL_BINARYOPERATION_H

#include <string>
#include <memory>
#include <utility>
#include "Operation.h"
#include "../utils.h"

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

#endif //FFITESTHELPER_BINARYOPERATION_H
