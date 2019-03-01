//
// Created by Radek on 27.02.2019.
//

#include "Assign.h"

Assign::Assign(std::shared_ptr<Variable> variable, std::shared_ptr<Output> value) {
    not_implemented<void>();
}

std::shared_ptr<Output> Assign::make_assign(std::shared_ptr<Variable> variable, std::shared_ptr<Output> value) {
    return not_implemented<std::shared_ptr<Output>>();
}

void Assign::add_to_graph(GraphSession &graph) {
    TF_Output val = value->add_to_graph(graph);
    graph.register_assignment(variable->get_name(), val);
}
