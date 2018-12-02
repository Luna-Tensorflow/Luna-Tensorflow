//
// Created by radeusgd on 01.12.18.
//
#include "utils.h"

size_t hash_combine(size_t x, size_t y) {
    x ^= y + 0x9e3779b9 + (x << 6u) + (x >> 2u);
    return x;
}

TFException::TFException(TF_Status* status) : status(status) {
}

TFException::~TFException() {
    TF_DeleteStatus(status);
}

const char* TFException::what() const noexcept {
    return TF_Message(status);
}

template<>
void run_with_status(std::function<void(TF_Status*)> f) {
    TF_Status *status = TF_NewStatus();
    f(status);
    if (TF_GetCode(status) != TF_OK) {
        throw TFException(status);
    }
    TF_DeleteStatus(status);
}