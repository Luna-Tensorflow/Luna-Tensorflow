//
// Created by Radek on 27.02.2019.
//

#include "Variable.h"
#include "../ops/Operation.h"

Variable::Variable(std::string& name, std::shared_ptr<Tensor> default_value)
    : name(name), default_value(default_value) {
        hash = std::hash<std::string>()(name);
        hash = hash_combine(hash, default_value->hashcode());
    }

void Variable::add_to_graph(GraphSession &graph)
{
    auto ptr = my_output.lock();
    if(ptr)
    {
        LOG_GRAPH(hash_log(), "[default_value_hash] " + std::to_string(default_value->hashcode()),
            "[default_value_shape] " + vec_to_string(default_value->shape()));
        TF_OperationDescription *desc = TF_NewOperation(graph.get_underlying(),
            "Placeholder", ("var_"+name+"_"+std::to_string(ptr->hashcode())).c_str());
        AttrType("dtype", default_value->getType()).set(desc);

        auto* op = run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, desc, std::placeholders::_1));

        TF_Output placeholder_out =
            {
                .oper = op,
                .index = 0
            };

        graph.register_output_hash(ptr->hashcode(), placeholder_out);
        graph.register_variable(name, default_value, placeholder_out);
    }
}

std::string Variable::get_name()
{
	return name;
}

std::pair<std::shared_ptr<Variable>, std::shared_ptr<Output>> Variable::make_variable(std::string& name,
    std::shared_ptr<Tensor> default_value)
{
    auto variable = std::shared_ptr<Variable>(new Variable(name, default_value));
    auto output = std::make_shared<Output>(variable, 0);

    variable->my_output = output;

    return std::make_pair(variable, output);
}

size_t Variable::hashcode() const {
    return hash;
}

std::string Variable::hash_log() const {
    return "Variable: " + std::to_string(hash);
}