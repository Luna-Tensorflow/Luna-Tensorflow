//
// Created by Radek on 03.11.2018.
//

#ifndef TF_EXAMPLE_TENSOR_H
#define TF_EXAMPLE_TENSOR_H

#include <vector>
#include <tensorflow/c/c_api.h>

#include "TypeLabel.h"

class Tensor {
private:
    TF_Tensor *underlying;

public:
    Tensor(const std::vector<int64_t> &dims, TF_DataType DataTypeLabel);

    Tensor(TF_Tensor* underlying);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other);

    template<TF_DataType DataTypeLabel>
    typename Type<DataTypeLabel>::type& at(const std::vector<int64_t> &indices) {
        int64_t index = indices.back();
        int64_t multiplier = 1;

        for (int i = indices.size() - 2; i >= 0; --i) {
            multiplier *= TF_Dim(underlying, i + 1);
            index += indices[i] * multiplier;
        }

        char* adr = (char*) TF_TensorData(underlying) + TF_DataTypeSize(DataTypeLabel) * index;

        return *(typename Type<DataTypeLabel>::type*)adr;
    }

    TF_Tensor* get_underlying() const;

    ~Tensor();
};


#endif //TF_EXAMPLE_TENSOR_H
