//
// Created by mateusz on 04.12.18.
//

#ifndef TFL_GRAPH_H
#define TFL_GRAPH_H

#include <map>
#include <tensorflow/c/c_api.h>
#include <memory>
#include <string>

#include "../helpers/utils.h"
#include "../tensor/Tensor.h"
#include "../helpers/LifeTimeManager.h"

#include <any>

template<TF_DataType DataTypeLabel> class Operation;
template<TF_DataType DataTypeLabel> class Placeholder;

class GraphSession
{
private:
	std::map<size_t, TF_Output> hashes;
	TF_Graph* graph;
	TF_Session* session;
	TF_SessionOptions* options;

	std::vector<TF_Output> output_nodes;
	std::map<std::string, TF_Output> placeholders;


public:
	GraphSession();
	~GraphSession();

	template<TF_DataType DataTypeLabel> bool exists(const Operation<DataTypeLabel>* op) {
		return (hashes.find(op->hashcode()) != hashes.end());
	}

	template<TF_DataType DataTypeLabel> TF_Output add_operation(const Operation<DataTypeLabel>* op)
	{
		if(exists(op))
			return hashes[op->hashcode()];

		return (hashes[op->hashcode()] = op->add_to_graph(*this));
	}

	template<TF_DataType DataTypeLabel>
	Tensor<DataTypeLabel>** eval(const std::map<std::string, std::shared_ptr<Tensor<DataTypeLabel>>>& substitutions) const
	{
		size_t count = output_nodes.size();
		std::vector<TF_Tensor*> output_values(count);

		if(!std::equal(placeholders.begin(), placeholders.end(), substitutions.begin(),
			[](auto& a, auto& b) -> bool {return a.first == b.first; }))
		{
			throw std::invalid_argument("Not all placeholders are substituted!");
		}

		std::vector<TF_Output> placeholders_v;
		std::vector<TF_Tensor*> tensor_v;

		for(auto elem : substitutions)
		{
			placeholders_v.push_back(placeholders.at(elem.first));
			tensor_v.push_back(elem.second->get_underlying());
		}

		run_with_status<void>(std::bind(TF_SessionRun,
		                                session,
		                                nullptr,
		                                placeholders_v.data(), tensor_v.data(), tensor_v.size(),
		                                output_nodes.data(), output_values.data(), count,
		                                nullptr, 0,
		                                nullptr,
		                                std::placeholders::_1));

		auto return_values = (Tensor<DataTypeLabel>**) std::calloc(sizeof(Tensor<DataTypeLabel>*), count);

		for(unsigned i=0; i<count; ++i)
		{
			return_values[i] = LifetimeManager::instance().addOwnership(std::make_shared<Tensor<DataTypeLabel>>(output_values[i]));
		}

		return return_values;
	}

	template<TF_DataType DataTypeLabel>
	Tensor<DataTypeLabel>** eval() const
	{
		return eval<DataTypeLabel>(std::map<std::string, std::shared_ptr<Tensor<DataTypeLabel>>>());
	}

	void register_output_hash(size_t hash, TF_Output &out) {
		hashes[hash] = out;
	}

	void register_placeholder(const std::string& name, TF_Output &out)
	{
		placeholders.emplace(name, out);
	}

	void add_output(TF_Output out)
	{
		output_nodes.push_back(out);
	}

	TF_Graph* get_underlying() {
		return graph;
	}

	TF_Session* get_underlying_session() {
		return session;
	}
};


#endif //TFL_GRAPH_HPP
