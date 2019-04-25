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
        FFILOG(height, width);
        int64_t dims[] = {height, width, 3};
        std::shared_ptr<Tensor> retval = std::make_shared<Tensor>(dims, 3, TF_FLOAT);

        for(int w=0; w<width; ++w) {
            for(int h=0; h<height; ++h) {
                auto pixl = image.get_pixel(w, h);
                retval->at<TF_FLOAT>(std::vector<int64_t >{h,w,0}) = pixl.red / 255.f;
                retval->at<TF_FLOAT>(std::vector<int64_t >{h,w,1}) = pixl.green / 255.f;
                retval->at<TF_FLOAT>(std::vector<int64_t >{h,w,2}) = pixl.blue / 255.f;
            }
        }



        return LifetimeManager::instance().addOwnership(std::move(retval));
    };
}