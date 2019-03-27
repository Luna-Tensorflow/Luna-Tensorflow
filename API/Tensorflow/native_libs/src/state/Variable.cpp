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
        LOG_GRAPH(hash_log(), "[name]", name, "[default_value_hash] " + std::to_string(default_value->hashcode()),
            "[default_value_shape] " + vec_to_string(default_value->shape()));

        TF_OperationDescription *desc = TF_NewOperation(graph.get_underlying(),
            "Variable", ("var_"+name+"_"+std::to_string(ptr->hashcode())).c_str());
        AttrType("dtype", default_value->getType()).set(desc);
        AttrShape("shape", default_value->shape()).set(desc);

        auto* op = run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, desc, std::placeholders::_1));

        TF_Output var_out =
            {
                .oper = op,
                .index = 0
            };

        TF_OperationDescription *placeholder_desc = TF_NewOperation(graph.get_underlying(),
                                                        "Placeholder", ("var_initp_"+name+"_"+std::to_string(ptr->hashcode())).c_str());
        AttrType("dtype", default_value->getType()).set(placeholder_desc);
        auto* placeholder = run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, placeholder_desc, std::placeholders::_1));
        TF_Output initializerPlaceholder = {
            .oper = placeholder,
            .index = 0
        };

        TF_OperationDescription *assign_desc = TF_NewOperation(graph.get_underlying(),
                                                                    "Assign", ("var_inita_"+name+"_"+std::to_string(ptr->hashcode())).c_str());

        TF_AddInput(assign_desc, var_out);
        TF_AddInput(assign_desc, initializerPlaceholder);

        auto* initializerAssign = run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, assign_desc, std::placeholders::_1));


        graph.register_output_hash(ptr->hashcode(), var_out);
        graph.register_variable(name, VariableDesc{
            .output = var_out,
            .initializerAssign = initializerAssign,
            .initializerPlaceholder = initializerPlaceholder,
            .default_value = default_value
        });
    }
}

std::string Variable::get_name()
{
	return name;
}

std::shared_ptr<Output> Variable::make_variable(std::string& name,
    std::shared_ptr<Tensor> default_value)
{
    auto variable = std::shared_ptr<Variable>(new Variable(name, default_value));
    auto output = std::make_shared<Output>(variable, 0);

    variable->my_output = output;

    return output;
}

size_t Variable::hashcode() const {
    return hash;
}

std::string Variable::hash_log() const {
    return "Variable: " + std::to_string(hash);
}