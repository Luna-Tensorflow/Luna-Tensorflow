#ifndef TFL_TYPELABEL_H
#define TFL_TYPELABEL_H

#include <tensorflow/c/c_api.h>

template<TF_DataType DataTypeLabel>
class Type {
};

template<typename type>
class SimpleType {
public:
    using tftype = type;
    using tfattype = type&;
    using lunatype = type;
};

#define SIMPLETYPE(label, type) template<> class Type<label> { \
public: \
    using tftype = type; \
    using tfattype = type&; \
    using lunatype = type; \
}

SIMPLETYPE(TF_FLOAT, float);
SIMPLETYPE(TF_DOUBLE, double);
SIMPLETYPE(TF_INT8, int8_t);
SIMPLETYPE(TF_INT16, int16_t);
SIMPLETYPE(TF_INT32, int32_t);
SIMPLETYPE(TF_INT64, int64_t);
SIMPLETYPE(TF_UINT8, int8_t);
SIMPLETYPE(TF_UINT16, uint16_t);
SIMPLETYPE(TF_UINT32, uint32_t);
SIMPLETYPE(TF_UINT64, uint64_t);

template<>
class Type<TF_BOOL> {
public:
    using tftype = bool;
    using tfattype = bool&;
    using lunatype = uint8_t;
};

template<>
class Type<TF_STRING> {
public:
    using tftype = const char*;
    using tfattype = char*;
    using lunatype = char*;
};

#endif //TFL_TYPELABEL_H
