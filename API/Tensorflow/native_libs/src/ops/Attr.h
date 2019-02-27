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

class AttrTypeList : public Attr {
private:
    std::vector<TF_DataType> types;

public:
    AttrTypeList(const std::string &name, const std::vector<TF_DataType> &types);

    void set(TF_OperationDescription *desc) const;
};

class AttrShape : public Attr {
private:
    std::vector<int64_t> dims;

public:
    AttrShape(const std::string &name, const std::vector<int64_t> &dims);

    void set(TF_OperationDescription *desc) const;
};

class AttrShapeList : public Attr {
private:
    std::vector<std::vector<int64_t>> dims;

public:
    AttrShapeList(const std::string &name, std::vector<std::vector<int64_t>> dims);

    void set(TF_OperationDescription *desc) const;
};

class AttrTensor : public Attr {
private:
    Tensor tensor;

public:
    AttrTensor(const std::string &name, const Tensor &tensor);

    void set(TF_OperationDescription *desc) const;
};

class AttrTensorList : public Attr {
private:
    std::vector<Tensor> tensors;

public:
    AttrTensorList(const std::string &name, const std::vector<Tensor> &tensors);

    void set(TF_OperationDescription *desc) const;
};

class AttrInt : public Attr {
private:
    int64_t value;

public:
    AttrInt(const std::string &name, int64_t value);

    void set(TF_OperationDescription *desc) const;
};

class AttrIntList : public Attr {
private:
    std::vector<int64_t> values;

public:
    AttrIntList(const std::string &name, const std::vector<int64_t> &values);

    void set(TF_OperationDescription *desc) const;
};

class AttrFloat : public Attr {
private:
    float value;

public:
    AttrFloat(const std::string &name, float value);

    void set(TF_OperationDescription *desc) const;
};

class AttrFloatList : public Attr {
private:
    std::vector<float> values;

public:
    AttrFloatList(const std::string &name, const std::vector<float> &values);

    void set (TF_OperationDescription *desc) const;
};

class AttrBool : public Attr {
private:
    bool value;

public:
    AttrBool(const std::string &name, bool value);

    void set(TF_OperationDescription *desc) const;
};

class AttrBoolList : public Attr {
private:
    std::vector<bool> values;

public:
    AttrBoolList(const std::string &name, const std::vector<bool> &values);

    void set(TF_OperationDescription *desc) const;
};

class AttrString : public Attr {
private:
    std::string value;

public:
    AttrString(const std::string &name, const std::string &value);

    void set(TF_OperationDescription *desc) const;
};

class AttrStringList : public Attr {
private:
    std::vector<std::string> values;

public:
    AttrStringList(const std::string& name, const std::vector<std::string> &values);

    void set(TF_OperationDescription *desc) const;
};

class AttrFuncName : public Attr {
private:
    std::string func_name;

public:
    AttrFuncName(const std::string &name, const std::string &func_name);

    void set(TF_OperationDescription *desc) const;
};

#endif //TFL_ATTR_H
