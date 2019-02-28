//
// Created by mateusz on 04.12.18.
//
#include <cstdlib>

#include "GraphSession.h"

GraphSession::GraphSession() {
    char suppress_tf_log[] = "TF_CPP_MIN_LOG_LEVEL=3";
    putenv(suppress_tf_log);

    graph = TF_NewGraph();
    options = TF_NewSessionOptions();
    session = run_with_status<TF_Session *>(std::bind(TF_NewSession, graph, options, std::placeholders::_1));
}

GraphSession::~GraphSession() {
    run_with_status<void>(std::bind(TF_DeleteSession, session, std::placeholders::_1));
    TF_DeleteSessionOptions(options);
    TF_DeleteGraph(graph);
}

bool GraphSession::exists(const Output *out) {
    return (hashes.find(out->hashcode()) != hashes.end());
}

TF_Output GraphSession::add_output(const Output *out) {
    if (exists(out))
        return hashes[out->hashcode()];

    return (hashes[out->hashcode()] = out->add_to_graph(*this));
}

std::shared_ptr<EvaluationResult> GraphSession::eval(const std::map<std::string, std::shared_ptr<Tensor>> &substitutions, const std::shared_ptr<State>& state) const {
    for (auto &ph : placeholders) {
        if (substitutions.count(ph.first) > 0)
            continue;
        // TODO maybe only print what's missing
        std::string err;
        err += "Not all placeholders are substituted!\n";
        err += "Placeholders: ";
        for (const auto &kv : placeholders) {
            err += kv.first + ", ";
        }
        err += "\n";
        err += "Substitutions: ";
        for (const auto &kv : substitutions) {
            err += kv.first + ", ";
        }
        err += "\n";

        throw std::invalid_argument(err);
    }

    std::vector<TF_Output> placeholders_v;
    std::vector<TF_Tensor *> tensor_v;

    for (auto elem : substitutions) {
        if (placeholders.find(elem.first) == placeholders.end()) //bypass obsolete substs
            continue;
        placeholders_v.push_back(placeholders.at(elem.first));
        tensor_v.push_back(elem.second->get_underlying());
    }

    std::vector<TF_Output> computed_outs = output_nodes;
    for (auto & p : assignments) {
        computed_outs.push_back(p.second);
    }
    std::vector<TF_Tensor *> output_values(computed_outs.size());

    run_with_status<void>(std::bind(TF_SessionRun,
                                    session,
                                    nullptr,
                                    placeholders_v.data(), tensor_v.data(), tensor_v.size(),
                                    computed_outs.data(), output_values.data(), computed_outs.size(),
                                    nullptr, 0,
                                    nullptr,
                                    std::placeholders::_1));

    auto r = std::make_shared<EvaluationResult>();

    // prepare output tensors
    r->outputs.resize(output_nodes.size());
    for (unsigned i = 0; i < output_nodes.size(); ++i) {
        r->outputs[i] = std::make_shared<Tensor>(output_values[i]);
    }

    // update state
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> updates;
    size_t idx = output_nodes.size();
    for (auto & p : assignments) {
        updates.emplace_back(p.first, std::make_shared<Tensor>(output_values[idx]));
        idx += 1;
    }

    r->result_state = state->updated(updates);

    return r;
}

std::vector<std::shared_ptr<Tensor>> GraphSession::eval() const {
    return eval(std::map<std::string, std::shared_ptr<Tensor>>(), State::make_empty())->outputs;
}

void GraphSession::register_output_hash(size_t hash, TF_Output &out) {
    hashes[hash] = out;
}

void GraphSession::register_placeholder(const std::string &name, TF_Output &out) {
    placeholders.emplace(name, out);
}

void GraphSession::add_fetched_output(TF_Output out) {
    output_nodes.push_back(out);
}

TF_Graph *GraphSession::get_underlying() {
    return graph;
}

TF_Session *GraphSession::get_underlying_session() {
    return session;
}

void GraphSession::register_assignment(const std::string &name, TF_Output value) {
    assignments[name] = value;
}
