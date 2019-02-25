//
// Created by wojtek on 13.02.19.
//

#ifndef TFL_ATTR_H
#define TFL_ATTR_H

#include <string>
#include <cstddef>

#include "tensorflow/c/c_api.h"

#include "../tensor/Tensor.h"

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

class AttrInt : public Attr {
private:
    int64_t value;

public:
    AttrInt(const std::string &name, int64_t value);

    void set(TF_OperationDescription *desc) const;
};

class AttrFloat : public Attr {
private:
    float value;

public:
    AttrFloat(const std::string &name, float value);

    void set(TF_OperationDescription *desc) const;
};

class AttrBool : public Attr {
private:
    bool value;

public:
    AttrBool(const std::string &name, bool value);

    void set(TF_OperationDescription *desc) const;
};

class AttrString : public Attr {
private:
    std::string value;

public:
    AttrString(const std::string &name, const std::string &value);

    void set(TF_OperationDescription *desc) const;
};

#endif //TFL_ATTR_H
