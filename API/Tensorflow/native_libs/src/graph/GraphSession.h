//
// Created by mateusz on 04.12.18.
//

#ifndef TFL_GRAPH_H
#define TFL_GRAPH_H

#include <map>
#include <tensorflow/c/c_api.h>
#include <memory>

#include "../helpers/utils.h"

template<TF_DataType DataTypeLabel>
class Operation;

class GraphSession
{
private:
	std::map<size_t, TF_Output> hashes;
	TF_Graph* graph;
	TF_Session* session;
	TF_SessionOptions* options;

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

	template<TF_DataType DataTypeLabel> TF_Output add(const Operation<DataTypeLabel>* op) {
		if(exists(op))
			return hashes[op->hashcode()];

		return (hashes[op->hashcode()] = op->add_to_graph(*this));
	}

	void register_output(size_t hash, TF_Output& out) {
		hashes[hash] = out;
	}

	TF_Graph* get_underlying() {
		return graph;
	}

	TF_Session* get_underlying_session() {
		return session;
	}
};


#endif //TFL_GRAPH_HPP
