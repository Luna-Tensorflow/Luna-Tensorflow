//
// Created by mateusz on 13.12.18.
//

#ifndef TFL_PLACEHOLDER_H
#define TFL_PLACEHOLDER_H

#include "Operation.h"
#include <string>

template<TF_DataType DataTypeLabel>
class Placeholder : public Operation<DataTypeLabel>
{
public:
	explicit Placeholder(std::string name)
		: operation_name(std::move(name)) {
		hash = std::hash<std::string>()("Placeholder "+operation_name);
	}

	size_t hashcode() const override {
		return hash;
	}

	TF_Output add_to_graph(GraphSession& graph) const override
	{
		if(graph.exists(this))
			return graph.add_operation(this);

		TF_OperationDescription *desc = TF_NewOperation(graph.get_underlying(),
		                                                "Placeholder", std::to_string(hashcode()).c_str());

		TF_SetAttrType(desc, "dtype", DataTypeLabel);

		TF_Operation *operation = run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, desc, std::placeholders::_1));

		TF_Output out =
			{
				.oper = operation,
				.index = 0
			};

		graph.register_output_hash(hashcode(), out);
		graph.register_placeholder(operation_name, out);
		return out;
	}

	std::string get_name() const
	{
		return operation_name;
	}

protected:
	size_t hash;
	std::string operation_name;
};


#endif //TFL_PLACEHOLDER_HPP
