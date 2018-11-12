#include "TFException.h"

TFException::TFException(TF_Status* status) : status(status) {
}

TFException::~TFException() {
    TF_DeleteStatus(status);
}

const char* TFException::what() const noexcept {
    return TF_Message(status);
}

void delete_or_throw(TF_Status* status) {
    if (TF_GetCode(status) == TF_OK) {
        TF_DeleteStatus(status);
    } else {
        throw TFException(status);
    }
}