//
// Created by wojtek on 13.02.19.
//

#include "Operation.h"
#include "../helpers/salter.h"

Operation::Operation(std::string name, std::vector<std::shared_ptr<Output>> inputs,
        std::vector<std::shared_ptr<Attr>> attrs, std::string chosen_name)
        : name(name), inputs(inputs), attrs(attrs), chosen_name(chosen_name) {

    hash = std::hash<std::string>()(name);
    if (name == "Const") {
        hash = hash_combine(hash, const_salter.next_salt());
    }
    hash = hash_combine(hash, std::hash<std::string>()(chosen_name));
    for(auto &input : inputs) {
        hash = hash_combine(hash, input->hashcode());
    }
    for(auto &attr : attrs) {
        hash = hash_combine(hash, attr->hashcode());
    }

    if (this->chosen_name.empty()) {
        this->chosen_name = this->name + std::to_string(hash);
    }
}

std::vector<std::shared_ptr<Output>> Operation::make_operation(std::string name,
                                                               std::vector<std::shared_ptr<Output>> inputs,
                                                               int num_outputs,
                                                               std::vector<std::shared_ptr<Attr>> attrs,
                                                               std::string chosen_name) {
    // TODO cannot use make_shared because constructor is private, it would be good to do something about this
    auto operation = std::shared_ptr<Operation>(new Operation(name, inputs, attrs, chosen_name));

    std::vector<std::shared_ptr<Output>> ret;
    for (int i = 0; i < num_outputs; ++i) {
        auto out = std::make_shared<Output>(operation, i);
        operation->outputs.emplace_back(out); // we construct a weak pointer to the output
        ret.push_back(out);
    }

    return ret;
}

void Operation::add_to_graph(GraphSession &graph) {
    std::vector<TF_Output> tf_inputs;
    std::string inputs_hashes;

    for (auto &input : inputs) {
        tf_inputs.push_back(graph.add_output(input.get()));
        inputs_hashes += input->get_binder()->hash_log() + " ";
    }

    LOG_GRAPH(hash_log(), "[chosen_name] "+chosen_name, "[inputs] " + inputs_hashes);

    TF_OperationDescription *desc = TF_NewOperation(graph.get_underlying(),
                                                    name.c_str(), chosen_name.c_str());

    for (auto &tf_input : tf_inputs) {
        TF_AddInput(desc, tf_input);
    }

    for (auto &attr : attrs) {
        attr->set(desc);
    }

    auto *operation = run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, desc, std::placeholders::_1));

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto out = outputs[i].lock(); // promote to shared_ptr, will return null if pointer is already disposed of
        if (out) {
            TF_Output tf_output = {
                    .oper = operation,
                    .index = static_cast<int>(i)
            };
            graph.register_output_hash(out->hashcode(), tf_output);
        }
    }

    if (name == "Placeholder") {
        TF_Output tf_output = {
                .oper = operation,
                .index = 0
        };
        graph.register_placeholder(chosen_name, tf_output);
    }
}

size_t Operation::hashcode() const {
    return hash;
}

std::string Operation::hash_log() const {
    return name + ": " + std::to_string(hash);
}