//
// Created by mateusz on 13.12.18.
//

#ifndef TFL_PARTIAL_H
#define TFL_PARTIAL_H

#include <vector>

#include "BinaryOperation.h"
#include "../helpers/utils.h"

template <TF_DataType DataTypeLabel> class Gradient;

template<TF_DataType DataTypeLabel>
class Partial : public Operation<DataTypeLabel> {
public:
    Partial(std::shared_ptr<Operation<DataTypeLabel>> x, std::shared_ptr<Gradient<DataTypeLabel>> gradient, size_t hash_base) :
			hash(hash_combine(hash_base, x->hashcode())), x(std::move(x)), gradient(std::move(gradient)) {}


    TF_Output add_to_graph(GraphSession& graph) const override {
		if (!graph.exists(this)) {
			gradient->add_to_graph(graph);
		}

		return graph.add_operation(this);
    };

	size_t hashcode() const override {
		return hash;
	}

private:
    size_t hash;
    std::shared_ptr<Operation<DataTypeLabel>> x;
    std::shared_ptr<Gradient<DataTypeLabel>> gradient;
};


#endif //TFL_PARTIAL_H
