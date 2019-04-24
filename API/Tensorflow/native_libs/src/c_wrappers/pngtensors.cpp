//
// Created by mateusz on 24.04.19.
//

#include "pngtensors.h"
#include "common.h"

#include <tensorflow/c/c_api.h>

#include "../png/image.hpp"
#include "../tensor/Tensor.h"

#include <memory>
#include "../helpers/LifeTimeManager.h"
#include "../helpers/error.h"


TFL_API Tensor* read_tensor_from_png(const char* filename, const char** outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(std::string(filename));
        png::image <png::rgb_pixel> image(filename);

        int64_t width = image.get_width(), height = image.get_height();
        auto data = static_cast<float*>(malloc(width*height*3*sizeof(float)));

        for(int w=0; w<width; ++w) {
            for(int h=0; h<height; ++h) {
                auto pixl = image.get_pixel(w, h);
                data[3*w*h] = pixl.red / 255.f;
                data[3*w*h+1] = pixl.green / 255.f;
                data[3*w*h+2] = pixl.blue / 255.f;
            }
        }
        int64_t dims[] = {width, height, 3};
        std::shared_ptr<Tensor> retval = std::make_shared<Tensor>(data, dims, 3, TF_FLOAT);
        free(data);

        return LifetimeManager::instance().addOwnership(std::move(retval));
    };
}