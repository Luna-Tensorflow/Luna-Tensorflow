//
// Created by mateusz on 04.12.18.
//

#ifndef TFL_GRAPH_H
#define TFL_GRAPH_H

#include <map>
#include <tensorflow/c/c_api.h>
#include <memory>

#include "../helpers/utils.h"
#include "../tensor/Tensor.h"
#include "../helpers/LifeTimeManager.h"

template<TF_DataType DataTypeLabel>
class Operation;

class GraphSession
{
private:
	std::map<size_t, TF_Output> hashes;
	TF_Graph* graph;
	TF_Session* session;
	TF_SessionOptions* options;

	std::vector<TF_Output> outputs;

public:
	GraphSession() 	{
		graph = TF_NewGraph();
		options = TF_NewSessionOptions();
		session = run_with_status<TF_Session*>(std::bind(TF_NewSession, graph, options, std::placeholders::_1));
	}

	~GraphSession()	{
		run_with_status<void>(std::bind(TF_DeleteSession, session, std::placeholders::_1));
		TF_DeleteSessionOptions(options);
		TF_DeleteGraph(graph);
	}

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
	Tensor<DataTypeLabel>** eval() const
	{
		size_t count = outputs.size();
		std::vector<TF_Tensor*> output_values(count);

		run_with_status<void>(std::bind(TF_SessionRun,
		                                session,
		                                nullptr,
		                                nullptr, nullptr, 0,
		                                outputs.data(), output_values.data(), count,
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

	void register_output_hash(size_t hash, TF_Output &out) {
		hashes[hash] = out;
	}

	void add_output(TF_Output out)
	{
		outputs.push_back(out);
	}

	TF_Graph* get_underlying() {
		return graph;
	}

	TF_Session* get_underlying_session() {
		return session;
	}
};


#endif //TFL_GRAPH_HPP
