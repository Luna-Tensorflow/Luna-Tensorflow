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
    std::string name = "assign_on_"+variable->get_name()+"_"+std::to_string(hashcode); //TODO: some fancy name
    TF_OperationDescription *desc = TF_NewOperation(graph.get_underlying(),
        "Assign", name.c_str());

    auto ptr = variable->get_output().lock();
    if(!ptr) //variable not added to graph (is it possible?)
    {
        throw std::invalid_argument("Assign to nonexistent variable");
    }

    TF_Output ref = ptr->add_to_graph(graph);
    TF_Output val = value->add_to_graph(graph);

    TF_AddInput(desc, ref);
    TF_AddInput(desc, val);

    auto* operation =
        run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, desc, std::placeholders::_1));

    TF_Output output_ref = {
        .oper = operation,
        .index = 0
    };

    graph.register_assignment(name, output_ref);
}
