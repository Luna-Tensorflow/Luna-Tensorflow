#ifndef TF_EXAMPLE_TFEXCEPTION_H
#define TF_EXAMPLE_TFEXCEPTION_H

#include <stdexcept>
#include <tensorflow/c/c_api.h>

class TFException : public std::exception {
private:
    TF_Status* status;
public:
    // exception takes ownership of the status
    TFException(TF_Status* status);

    ~TFException() override;

    const char* what() const noexcept override;
};

void delete_or_throw(TF_Status* status);

#endif //TF_EXAMPLE_TFEXCEPTION_H
