//
// Created by wojtek on 20.12.18.
//

#ifndef TFL_GRADIENT_WRAPPER_H
#define TFL_GRADIENT_WRAPPER_H

#include <tensorflow/c/c_api.h>

#include "../ops/Operation.h"
#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

TFL_API Operation<TF_FLOAT>** add_gradient(Operation<TF_FLOAT>** ys, int nys, Operation<TF_FLOAT>** xs, int nxs,
        Operation<TF_FLOAT>** dxs);

#ifdef __cplusplus
}
#endif

#endif //TFL_GRADIENT_WRAPPER_H
