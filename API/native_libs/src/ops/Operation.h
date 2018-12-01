#ifndef FFITESTHELPER_OPERATION_H
#define FFITESTHELPER_OPERATION_H


#include <cstddef>

class Operation {
public:
    virtual size_t hashcode() = 0;
    virtual ~Operation() = default;
};


#endif //FFITESTHELPER_OPERATION_H
