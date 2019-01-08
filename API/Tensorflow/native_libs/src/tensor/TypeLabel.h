#ifndef TFL_TYPELABEL_H
#define TFL_TYPELABEL_H

#include <tensorflow/c/c_api.h>

template<TF_DataType DataTypeLabel>
class Type {
};

template<>
class Type<TF_FLOAT> {
public:
	using type = float;
};

template<>
class Type<TF_DOUBLE> {
public:
	using type = double;
};


template<>
class Type<TF_INT8> {
public:
	using type = int8_t;
};

template<>
class Type<TF_INT16> {
public:
	using type = int16_t;
};

template<>
class Type<TF_INT32> {
public:
	using type = int32_t;
};

template<>
class Type<TF_INT64> {
public:
	 using type = int64_t;
};

template<>
class Type<TF_UINT8> {
public:
	 using type = uint8_t ;
};

template<>
class Type<TF_UINT16> {
public:
	 using type = uint16_t;
};

template<>
class Type<TF_UINT32> {
public:
	using type = uint32_t;
};

template<>
class Type<TF_UINT64> {
public:
	using type = uint64_t;
};


#endif //TFL_TYPELABEL_H
