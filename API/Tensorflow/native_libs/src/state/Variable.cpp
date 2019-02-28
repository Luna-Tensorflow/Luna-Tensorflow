//
// Created by Radek on 27.02.2019.
//

#include "Variable.h"

Variable::Variable(std::shared_ptr<Tensor> defaultValue, std::string name) {
    not_implemented<void>();
}

void Variable::add_to_graph(GraphSession &graph)
{

}

std::string Variable::get_name()
{
	return not_implemented<std::string>();
}
