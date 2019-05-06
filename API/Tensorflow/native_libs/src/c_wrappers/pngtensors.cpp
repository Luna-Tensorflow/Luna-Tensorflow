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

#include <filesystem>

namespace fs = std::filesystem;

//namespace
//{
    Tensor* read_tensor_from_png_noerror(const char* filename) {
        std::cerr<<"read_tensor_from_png_noerror "<<std::string(filename)<<std::endl;
        png::image <png::rgb_pixel> image(filename);

        int64_t width = image.get_width(), height = image.get_height();
        int64_t dims[] = {height, width, 3};
        std::shared_ptr<Tensor> retval = std::make_shared<Tensor>(dims, 3, TF_FLOAT);


        for(int w=0; w<width; ++w) {
            for(int h=0; h<height; ++h) {
                auto pixl = image.get_pixel(w, h);
                int64_t base = 3*w + 3*width*h;
                retval->at<TF_FLOAT>(base) = pixl.red / 255.f;
                retval->at<TF_FLOAT>(base+1) = pixl.green / 255.f;
                retval->at<TF_FLOAT>(base+2) = pixl.blue / 255.f;
            }
        }

        return LifetimeManager::instance().addOwnership(std::move(retval));
    }
//}

TFL_API Tensor* read_tensor_from_png(const char* filename, const char** outError) {
    return TRANSLATE_EXCEPTION(outError) {
        return read_tensor_from_png_noerror(filename);
    };
}

TFL_API int png_files_in_directory_count(const char* path, const char** outError) {
    return TRANSLATE_EXCEPTION(outError) {
    std::cerr<<"TU"<<std::endl;
        std::string dir(path);
        std::cerr<<"TU "<<dir<<std::endl;
        auto it = fs::directory_iterator(dir, fs::directory_options::skip_permission_denied);
        std::cerr<<"TU"<<std::endl;

        LOGANDRETURN(std::count_if(fs::begin(it), fs::end(it), [](auto& entry)
        {
            std::cerr<<"TU"<<std::endl;
            return (entry.path().extension() == ".png") || (entry.path().extension() == ".PNG");
        }), std::string(path));
    };
}

TFL_API Tensor** read_tensor_arr_from_png_directory(const char* path, const char** outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        std::string dir(path);
        int count = png_files_in_directory_count(path, outError);
        int idx = 0;
        std::cerr<<count<<std::endl;

        auto tensors = static_cast<Tensor**>(malloc(count*sizeof(Tensor*)));

        for(const auto& entry : fs::directory_iterator(dir))
        {
            if(entry.path().extension() == ".png" || entry.path().extension() == ".PNG")
            {
                tensors[idx++] = read_tensor_from_png_noerror(entry.path().extension().c_str());
            }
        }
        LOGANDRETURN(tensors, dir);
    };
}