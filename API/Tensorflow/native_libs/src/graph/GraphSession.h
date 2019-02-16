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
#include "../ops/Output.h"

#include <any>

class Output;

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

	bool exists(const Output* out);

	TF_Output add_output(const Output* out);

	Tensor** eval(const std::map<std::string, std::shared_ptr<Tensor>>& substitutions) const;

    Tensor** eval() const;

	void register_output_hash(size_t hash, TF_Output &out);

	void register_placeholder(const std::string& name, TF_Output &out);

	void add_fetched_output(TF_Output out);

	TF_Graph* get_underlying();

	TF_Session* get_underlying_session();
};


#endif //TFL_GRAPH_HPP
