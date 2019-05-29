#ifndef TFL_GRAPH_H
#define TFL_GRAPH_H

#include <map>
#include <tensorflow/c/c_api.h>
#include <memory>
#include <string>

#include "../helpers/utils.h"
#include "../tensor/Tensor.h"
#include "../ops/Output.h"

#include <any>
#include <unordered_set>
#include "../state/State.h"

class Output;
class Variable;

struct EvaluationResult {
	 std::vector<std::shared_ptr<Tensor>> outputs;
	 std::shared_ptr<State> result_state;
};

struct VariableDesc {
	 TF_Output output;
	 TF_Operation* initializerAssign;
	 TF_Output initializerPlaceholder;
	 std::shared_ptr<Tensor> default_value;
};

class GraphSession
{
private:
	std::map<size_t, TF_Output> hashes;
	TF_Graph* graph;
	TF_Session* session;
	TF_SessionOptions* options;

	std::vector<TF_Output> fetched_output_nodes;
	std::map<std::string, TF_Output> placeholders;

	std::map<std::string, VariableDesc> variables;
	std::unordered_set<TF_Operation*> side_effects;
public:
    GraphSession();

    ~GraphSession();

    bool exists(const Output* out);
    TF_Output add_output(const Output* out);
    TF_Output get_output(const Output* out);

    std::shared_ptr<EvaluationResult> eval(const std::map<std::string, std::shared_ptr<Tensor>>& substitutions,
		const std::shared_ptr<State>& state = State::make_empty(), bool apply_side_effects = true) const;

    std::shared_ptr<EvaluationResult> eval(const std::shared_ptr<State>& state = State::make_empty()) const;
	void register_output_hash(size_t hash, TF_Output &out);


    void register_placeholder(const std::string& name, TF_Output &out);

    void add_fetched_output(TF_Output out);

    void register_sideefect(TF_Operation* effect);
	void register_variable(const std::string& name, VariableDesc variableDesc);

    void initialize_variables(const std::shared_ptr<State>& state) const; // this function is const as it doesnt modify the graph, despite modifying the mutable state, but it's used inside of eval which itself is rightfully const

    std::vector<std::shared_ptr<Tensor>> eval_one_step(const std::map<std::string, std::shared_ptr<Tensor>>& substitutions,
            bool apply_side_effects = true) const;

    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> read_variables() const; // TODO mark what's been updated

	TF_Graph* get_underlying();

	TF_Session* get_underlying_session();
};


#endif //TFL_GRAPH_HPP
