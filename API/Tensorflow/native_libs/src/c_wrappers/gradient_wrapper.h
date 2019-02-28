//
// Created by wojtek on 20.12.18.
//

#ifndef TFL_GRADIENT_WRAPPER_H
#define TFL_GRADIENT_WRAPPER_H

#include <cstdint>
#include <tensorflow/c/c_api.h>

#include "../ops/Operation.h"
#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define DEFINE_GRADIENT(typelabel) \
TFL_API Operation<typelabel>** add_gradients_##typelabel(Operation<typelabel>** ys, std::int64_t nys, Operation<typelabel>** xs, \
        std::int64_t nxs, Operation<typelabel>** dxs);

DEFINE_GRADIENT(TF_FLOAT);
DEFINE_GRADIENT(TF_DOUBLE);
DEFINE_GRADIENT(TF_INT8);
DEFINE_GRADIENT(TF_INT16);
DEFINE_GRADIENT(TF_INT32);
DEFINE_GRADIENT(TF_INT64);
DEFINE_GRADIENT(TF_UINT8);
DEFINE_GRADIENT(TF_UINT16);
DEFINE_GRADIENT(TF_UINT32);
DEFINE_GRADIENT(TF_UINT64);
DEFINE_GRADIENT(TF_BOOL);
//DEFINE_GRADIENT(TF_STRING);
//DEFINE_GRADIENT(TF_HALF);

#ifdef __cplusplus
}
#endif

#endif //TFL_GRADIENT_WRAPPER_H
