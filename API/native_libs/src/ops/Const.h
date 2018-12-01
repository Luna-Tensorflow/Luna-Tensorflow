#ifndef FFITESTHELPER_CONST_H
#define FFITESTHELPER_CONST_H


#include <API/native_libs/src/utils.h>
#include <memory>
#include <utility>
#include "Operation.h"

class Const : public Operation {
public:
    explicit Const(std::shared_ptr<TODOType> tensor) : value(std::move(tensor)) {
        hash = 0xdeadbeef; // TODO
    }

    size_t hashcode() override {
        return hash;
    }
private:
    size_t hash;
    std::shared_ptr<TODOType> value;
};

extern "C" Operation* make_op_const(TODOType* tensor);

#endif //FFITESTHELPER_CONST_H
