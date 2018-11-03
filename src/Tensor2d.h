//
// Created by Radek on 03.11.2018.
//

#ifndef TF_EXAMPLE_TENSOR2D_H
#define TF_EXAMPLE_TENSOR2D_H

#include <tensorflow/c/c_api.h>

class Tensor2d {
public:
    using DataType = float;
    static const TF_DataType DataTypeLabel = TF_FLOAT;
    Tensor2d(int n, int m) {
        int64_t dims[] = {n, m};
        underlying = TF_AllocateTensor(DataTypeLabel, dims, 2, n * m * TF_DataTypeSize(DataTypeLabel));
    }

    Tensor2d(const Tensor2d& other) = delete;
    Tensor2d(Tensor2d&& other) {
        underlying = other.underlying;
        other.underlying = nullptr;
    }

    float& operator()(int x, int y) {
        int64_t index = x * TF_Dim(underlying, 1) + y; // TODO make sure this layout is correct
        char* adr = (char*) TF_TensorData(underlying) + TF_DataTypeSize(DataTypeLabel) * index;
        return *(float*)adr;
    }

    TF_Tensor* get_underlying() const {
        return underlying;
    }

    ~Tensor2d() {
        if (underlying != nullptr) {
            TF_DeleteTensor(underlying);
        }
        underlying = nullptr;
    }
private:
    TF_Tensor* underlying;

};


#endif //TF_EXAMPLE_TENSOR2D_H
