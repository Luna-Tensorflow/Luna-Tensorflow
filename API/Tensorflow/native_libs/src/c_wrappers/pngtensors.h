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
    TFL_API Tensor**  read_tensor_arr_from_png_directory(const char* path, const char** outError);
    TFL_API int     png_files_in_directory_count(const char* path, const char** outError);
};

#endif //TFL_PNGTENSORS_H
