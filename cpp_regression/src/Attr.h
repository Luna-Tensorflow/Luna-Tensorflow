#ifndef TF_EXAMPLE_ATTR_H
#define TF_EXAMPLE_ATTR_H

#include <string>
#include <vector>
#include <tensorflow/c/c_api.h>

#include "Tensor.h"
#include "TFException.h"

class Attr {
protected:
    std::string name;

    Attr(const std::string &name);

public:
    virtual ~Attr() = default;

    virtual void set(TF_OperationDescription *desc) const = 0;
};

class AttrType : public Attr {
private:
    TF_DataType type;

public:
    AttrType(const std::string &name, TF_DataType type);

    void set(TF_OperationDescription *desc) const;
};

class AttrShape : public Attr {
private:
    std::vector<int64_t> dims;

public:
    AttrShape(const std::string &name, const std::vector<int64_t> &dims);

    void set(TF_OperationDescription *desc) const;
};

class AttrTensor : public Attr {
private:
    Tensor tensor;

public:
    AttrTensor(const std::string &name, const Tensor &tensor);

    void set(TF_OperationDescription *desc) const;
};

#endif //TF_EXAMPLE_ATTR_H
