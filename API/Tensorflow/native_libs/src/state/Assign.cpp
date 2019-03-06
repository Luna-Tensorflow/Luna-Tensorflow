//
// Created by Radek on 27.02.2019.
//

#include "Assign.h"

Assign::Assign(std::shared_ptr<Output> unit,
    std::shared_ptr<Variable> variable, std::shared_ptr<Output> value)
    : unit(unit), variable(variable), value(value) {
        hash = unit->hashcode();
        hash = hash_combine(hash, variable->hashcode());
        hash = hash_combine(hash, value->hashcode());
    }

std::shared_ptr<Output> Assign::make_assign(std::shared_ptr<Output> unit, std::shared_ptr<Variable> variable, std::shared_ptr<Output> value) {
    auto assignment = std::shared_ptr<Assign>(new Assign(unit, variable, value));

    auto out = std::make_shared<Output>(assignment, 0);
    assignment->output = out;

    return out;
}

void Assign::add_to_graph(GraphSession &graph) {
    // register the side effect of assignment
    TF_Output val = value->add_to_graph(graph);
    graph.register_assignment(variable->get_name(), val);

    auto out = output.lock();
    if (out) {
        // add input unit to graph
        TF_Output unitout = unit->add_to_graph(graph);

        // register our output's hash to point to the same output that our unit gave
        graph.register_output_hash(out->hashcode(), unitout);
    } else {
        // this will never happen, because Assign cannot outlive its Output
        throw std::logic_error("Causality violation");
    }
}

size_t Assign::hashcode() const {
    return hash;
} 
