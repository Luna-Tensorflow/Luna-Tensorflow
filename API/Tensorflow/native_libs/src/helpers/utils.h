#ifndef TFL_UTILS_H
#define TFL_UTILS_H

#include <cstddef>
#include <stdexcept>
#include <functional>
#include <tensorflow/c/c_api.h>

/*
 * A placeholder type used to mark places where other classes are not yet available.
 */
class TODOType {};

template <typename T> T not_implemented() {
    throw std::logic_error("Not implemented");
}
/*
 * Based on https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
 */
size_t hash_combine(size_t x, size_t y);

class TFException : public std::exception {
private:
    TF_Status* status;
public:
    // exception takes ownership of the status
    TFException(TF_Status* status);

    ~TFException() override;

    const char* what() const noexcept override;
};

template<typename T>
T run_with_status(std::function<T(TF_Status*)> f) {
    TF_Status *status = TF_NewStatus();
    T ret = f(status);
    if (TF_GetCode(status) != TF_OK) {
        throw TFException(status);
    }
    TF_DeleteStatus(status);
    return ret;
}

template<>
void run_with_status(std::function<void(TF_Status*)> f);

#endif //TFL_UTILS_H
