//
// Created by mateusz on 13.12.18.
//

#ifndef TFL_GRADIENT_H
#define TFL_GRADIENT_H

#include "BinaryOperation.h"

template<TF_DataType DataTypeLabel>
class Gradient : public BinaryOperation<DataTypeLabel> {
public:
    Gradient(std::shared_ptr<Operation<DataTypeLabel>> a,
             std::shared_ptr<Operation<DataTypeLabel>> b)
        : BinaryOperation<DataTypeLabel>("Gradient", a, b) {}


    TF_Output add_to_graph(GraphSession& graph) const override {
        TF_Output output1 = graph.add_operation(this->arg1.get());
        TF_Output output2 = graph.add_operation(this->arg2.get());


		TF_Output output;
		run_with_status<void>(std::bind(TF_AddGradients, graph.get_underlying(),
		                           &output1, 1, &output2, 1, nullptr, std::placeholders::_1, &output));
	    graph.register_output_hash(this->hashcode(), output);

		return output;
    };
};


#endif //TFL_GRADIENT_H
