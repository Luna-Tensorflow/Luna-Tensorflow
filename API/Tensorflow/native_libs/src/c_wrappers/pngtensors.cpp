#include "pngtensors.h"
#include "common.h"

#include "../png/image.hpp"
#include "../tensor/Tensor.h"
#include "../helpers/LifeTimeManager.h"
#include "../helpers/error.h"

#include <tensorflow/c/c_api.h>
#include <memory>
#include <dirent.h>

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
        FFILOG(filename);
        return read_tensor_from_png_noerror(filename);
    };
}

std::vector<std::string> png_files_in_directory(const char* path)
{
    std::vector<std::string> result;

    dirent* de;
    DIR* dir = opendir(path);
    if (dir == nullptr)
        return result;

    while((de = readdir(dir)) != nullptr)
    {
        char* name = de->d_name;
        if(strlen(name) > 4 && !strcmp(name + strlen(name) - 4, ".png"))
        {
            std::string str(path);
            str.append("/");
            str.append(name);
            result.emplace_back(str);
        }
    }
    closedir(dir);
    return result;
}

TFL_API int png_files_in_directory_count(const char* path, const char** outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(path);
        return png_files_in_directory(path).size();
    };
}

TFL_API Tensor** read_tensor_arr_from_png_directory(const char* path, const char** outError)
{
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(path);
        auto files = png_files_in_directory(path);
        std::vector<Tensor*> tensors(files.size());
        std::transform(files.begin(), files.end(), tensors.begin(), [](std::string& path)
        {
            return read_tensor_from_png_noerror(path.c_str());
        });

        auto tensorsArr = static_cast<Tensor**>(malloc(tensors.size() * sizeof(Tensor*)));
        std::memcpy(tensorsArr, tensors.data(), tensors.size() * sizeof(Tensor*));

        return tensorsArr;
    };
}