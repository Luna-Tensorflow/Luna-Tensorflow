#include <cstdlib>
#include "../state/Variable.h"

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
        return get_output(out);

    return (hashes[out->hashcode()] = out->add_to_graph(*this));
}

TF_Output GraphSession::get_output(const Output *out) {
    return hashes[out->hashcode()];
}

std::shared_ptr<EvaluationResult> GraphSession::eval(
        const std::map<std::string, std::shared_ptr<Tensor>> &substitutions,
        const std::shared_ptr<State> &state,
        bool apply_side_effects) const {
    for (auto &ph : placeholders) {
        if (substitutions.count(ph.first) > 0) {
            continue;
        }

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

    initialize_variables(state);

    auto r = std::make_shared<EvaluationResult>();

    r->outputs = eval_one_step(substitutions, apply_side_effects);

    auto updates = read_variables();
    r->result_state = state->updated(updates);

    return r;
}

std::shared_ptr<EvaluationResult> GraphSession::eval(const std::shared_ptr<State> &state) const {
    return eval(std::map<std::string, std::shared_ptr<Tensor>>(), state);
}

void GraphSession::register_output_hash(size_t hash, TF_Output &out) {
    hashes[hash] = out;
}

void GraphSession::register_placeholder(const std::string &name, TF_Output &out) {
    placeholders.emplace(name, out);
}

void GraphSession::add_fetched_output(TF_Output out) {
    fetched_output_nodes.push_back(out);
}

TF_Graph *GraphSession::get_underlying() {
    return graph;
}

TF_Session *GraphSession::get_underlying_session() {
    return session;
}

void GraphSession::register_sideefect(TF_Operation *effect) {
    side_effects.insert(effect);
}

void GraphSession::register_variable(const std::string &name, VariableDesc variableDesc) {
    variables[name] = variableDesc;
}

void GraphSession::initialize_variables(const std::shared_ptr<State> &state) const {
    if(variables.empty()) //do nothing if there are no variables to initialize
        return;

    std::vector<TF_Output> placeholders;
    std::vector<TF_Tensor *> tensors;
    std::vector<TF_Operation *> assignments;

    for (auto &elem : variables) {
        const VariableDesc &vd = elem.second;
        const std::string &name = elem.first;

        LOG("vars", elem.first, vec_to_string(vd.default_value->shape()));

        auto value = state->get(name);
        if (!value) {
            value = vd.default_value;
        }

        placeholders.push_back(vd.initializerPlaceholder);
        assignments.push_back(vd.initializerAssign);
        tensors.push_back(value->get_underlying());
    }

    run_with_status<void>(std::bind(TF_SessionRun,
                                    session,
                                    nullptr,
                                    placeholders.data(), tensors.data(), tensors.size(),
                                    nullptr, nullptr, 0,
                                    assignments.data(), assignments.size(),
                                    nullptr,
                                    std::placeholders::_1));
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> GraphSession::read_variables() const {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> updates;

    if(!variables.empty()) //there is something to read
    {
        std::vector<TF_Output> computed_outs;
        std::vector<TF_Tensor *> output_values;

        for (auto elem : variables) {
            VariableDesc &vd = elem.second;
            computed_outs.push_back(vd.output);
            updates.emplace_back(elem.first, nullptr);
        }

        output_values.resize(computed_outs.size());

        run_with_status<void>(std::bind(TF_SessionRun,
                                        session,
                                        nullptr,
                                        nullptr, nullptr, 0,
                                        computed_outs.data(), output_values.data(), output_values.size(),
                                        nullptr, 0,
                                        nullptr,
                                        std::placeholders::_1));

        for (size_t i = 0; i < updates.size(); ++i) {
            updates[i].second = std::make_shared<Tensor>(output_values[i]);
        }
    }
    return updates;
}

std::vector<std::shared_ptr<Tensor>>
GraphSession::eval_one_step(const std::map<std::string, std::shared_ptr<Tensor>> &substitutions, bool apply_side_effects) const {
    std::vector<TF_Output> placeholders_vars_v;
    std::vector<TF_Tensor *> tensor_v;

    std::cerr<<"1" << std::endl;

    for (auto &elem : substitutions) {
        LOG("subs", elem.first, vec_to_string(elem.second->shape()));
        if (placeholders.find(elem.first) == placeholders.end()) //bypass obsolete substs
            continue;
        placeholders_vars_v.push_back(placeholders.at(elem.first));
        tensor_v.push_back(elem.second->get_underlying());
    }

    std::vector<TF_Output> computed_outs = fetched_output_nodes;

    std::vector<TF_Tensor *> output_values(computed_outs.size());

    std::vector<TF_Operation *> targets;
    if (apply_side_effects) {
        targets = std::vector<TF_Operation *>(side_effects.begin(), side_effects.end());
    }

    run_with_status<void>(std::bind(TF_SessionRun,
                                    session,
                                    nullptr,
                                    placeholders_vars_v.data(), tensor_v.data(), tensor_v.size(),
                                    computed_outs.data(), output_values.data(), output_values.size(),
                                    targets.data(), targets.size(),
                                    nullptr,
                                    std::placeholders::_1));

    std::vector<std::shared_ptr<Tensor>> outputs;
    // prepare output tensors
    outputs.resize(fetched_output_nodes.size());
    for (unsigned i = 0; i < fetched_output_nodes.size(); ++i) {
        outputs[i] = std::make_shared<Tensor>(output_values[i]);
    }
    return outputs;
}
