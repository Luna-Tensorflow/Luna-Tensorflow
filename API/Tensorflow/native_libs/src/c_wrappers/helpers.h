//
// Created by radeusgd on 05.03.19.
//

#ifndef TFL_HELPERS_H
#define TFL_HELPERS_H

#include "common.h"

extern "C" {
// this is a temporary workaround to easily generate stateful names in Luna until State monad is available
TFL_API int64_t get_and_increase_layer_counter(const char* name, const char **outError);
};

#endif //TFL_HELPERS_H
