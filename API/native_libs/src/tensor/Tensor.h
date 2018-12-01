#ifndef TFL_TENSOR_H
#define TFL_TENSOR_H

#include <vector>
#include <tensorflow/c/c_api.h>

#include "TypeLabel.h"

template<TF_DataType DataTypeLabel>
class Tensor {
private:
	TF_Tensor *underlying;
	using type = typename Type<DataTypeLabel>::type;
	std::vector<size_t> dims;

public:
	explicit Tensor(const type* array, size_t len);
	explicit Tensor(const type** array, size_t width, size_t height);
	explicit Tensor(const std::vector<type> &vect) : Tensor(vect.data(), vect.size()) {}
	explicit Tensor(const std::vector<std::vector<type>> &array);

	explicit Tensor(TF_Tensor* underlying);

	Tensor(const Tensor& other);

	Tensor(Tensor&& other) noexcept;

	type& at(const std::vector<int64_t> &indices);

	std::vector<size_t> shape();

	TF_Tensor* get_underlying() const;

	~Tensor();
};


#endif //TFL_TENSOR_H
