//
// Created by mateusz on 04.12.18.
//
#include <cstdlib>

#include "GraphSession.h"

GraphSession::GraphSession()
{
	char suppress_tf_log[] = "TF_CPP_MIN_LOG_LEVEL=3";
	putenv(suppress_tf_log);

	graph = TF_NewGraph();
	options = TF_NewSessionOptions();
	session = run_with_status<TF_Session*>(std::bind(TF_NewSession, graph, options, std::placeholders::_1));
}

GraphSession::~GraphSession()
{
	run_with_status<void>(std::bind(TF_DeleteSession, session, std::placeholders::_1));
	TF_DeleteSessionOptions(options);
	TF_DeleteGraph(graph);
}

bool GraphSession::exists(const Output* out) {
	return (hashes.find(out->hashcode()) != hashes.end());
}

TF_Output GraphSession::add_output(const Output* out)
{
	if(exists(out))
		return hashes[out->hashcode()];

	return (hashes[out->hashcode()] = out->add_to_graph(*this));
}

Tensor** GraphSession::eval(const std::map<std::string, std::shared_ptr<Tensor>>& substitutions) const
{
	size_t count = output_nodes.size();
	std::vector<TF_Tensor*> output_values(count);

	for(auto& ph : placeholders)
	{
		if(substitutions.count(ph.first) > 0)
			continue;
		// TODO maybe only print what's missing
		std::string err;
		err += "Not all placeholders are substituted!\n";
		err += "Placeholders: ";
		for (const auto& kv : placeholders) {
			err += kv.first + ", ";
		}
		err += "\n";
		err += "Substitutions: ";
		for (const auto& kv : substitutions) {
			err += kv.first + ", ";
		}
		err += "\n";

		throw std::invalid_argument(err);
	}

	std::vector<TF_Output> placeholders_v;
	std::vector<TF_Tensor*> tensor_v;

	for(auto elem : substitutions)
	{
		if(placeholders.find(elem.first) == placeholders.end()) //bypass obsolete substs
			continue;
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

	auto return_values = (Tensor**) std::malloc(sizeof(Tensor*) * count);

	for(unsigned i=0; i<count; ++i)
	{
		return_values[i] = LifetimeManager::instance().addOwnership(std::make_shared<Tensor>(output_values[i]));
	}

	return return_values;
}

Tensor** GraphSession::eval() const
{
	return eval(std::map<std::string, std::shared_ptr<Tensor>>());
}

void GraphSession::register_output_hash(size_t hash, TF_Output &out) {
	hashes[hash] = out;
}

void GraphSession::register_placeholder(const std::string& name, TF_Output &out)
{
	placeholders.emplace(name, out);
}

void GraphSession::add_fetched_output(TF_Output out)
{
	output_nodes.push_back(out);
}

TF_Graph* GraphSession::get_underlying() {
	return graph;
}

TF_Session* GraphSession::get_underlying_session() {
	return session;
}