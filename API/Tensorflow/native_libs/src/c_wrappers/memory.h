#ifndef TFL_MEMORY_H
#define TFL_MEMORY_H

#include "common.h"

extern "C" {
    TFL_API void release(void *handle) noexcept;
    TFL_API void free_pointer(void *pointer);
};

#endif //FFITESTHELPER_MEMORY_H
