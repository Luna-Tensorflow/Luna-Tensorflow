#ifndef TF_EXAMPLE_TYPELABEL_H
#define TF_EXAMPLE_TYPELABEL_H

#include <tensorflow/c/c_api.h>

template<TF_DataType DataTypeLabel>
class Type {
};

template<>
class Type<TF_FLOAT> {
public:
    using type = float;
};

#endif //TF_EXAMPLE_TYPELABEL_H
