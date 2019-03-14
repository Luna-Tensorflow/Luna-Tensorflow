//
// Created by wojtek on 20.12.18.
//

#ifndef TFL_GRADIENT_WRAPPER_H
#define TFL_GRADIENT_WRAPPER_H

#include <cstdint>
#include <tensorflow/c/c_api.h>

#include "../ops/Output.h"
#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

TFL_API Output** add_gradients(Output** ys, std::int64_t nys, Output** xs,
        std::int64_t nxs, Output** dxs);

#ifdef __cplusplus
}
#endif

#endif //TFL_GRADIENT_WRAPPER_H
