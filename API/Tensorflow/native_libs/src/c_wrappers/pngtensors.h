//
// Created by mateusz on 24.04.19.
//

#ifndef TFL_PNGTENSORS_H
#define TFL_PNGTENSORS_H

#include "common.h"
#include "../tensor/Tensor.h"

extern "C"
{
    TFL_API Tensor* read_tensor_from_png(const char* filename, const char** outError);

};

#endif //TFL_PNGTENSORS_H
