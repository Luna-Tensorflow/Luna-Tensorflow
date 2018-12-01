#ifndef TF_EXAMPLE_TENSOR_H
#define TF_EXAMPLE_TENSOR_H

#include <vector>
#include <tensorflow/c/c_api.h>

#include "TypeLabel.h"

template<TF_DataType DataTypeLabel>
class Tensor {
private:
	TF_Tensor *underlying;
	using type = typename Type<DataTypeLabel>::type;
	std::vector<size_t> dims;

	type& at(const std::vector<int64_t> &indices);
public:
	explicit Tensor(const std::vector<type> &vect);
	explicit Tensor(const std::vector<std::vector<type>> &array);

	explicit Tensor(TF_Tensor* underlying);

	Tensor(const Tensor& other);

	Tensor(Tensor&& other) noexcept;

	std::vector<size_t> shape();

	TF_Tensor* get_underlying() const;

	~Tensor();
};


#endif //TF_EXAMPLE_TENSOR_H
