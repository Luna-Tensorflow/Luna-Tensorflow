//
// Created by Radek on 27.02.2019.
//

#ifndef TFL_ASSIGN_H
#define TFL_ASSIGN_H

#include "../ops/Binder.h"
#include "Variable.h"

class Assign : public Binder {
public:
	static std::shared_ptr<Output> make_assign(std::shared_ptr<Output> unit, std::shared_ptr<Variable> variable, std::shared_ptr<Output> value);

    void add_to_graph(GraphSession &graph) override;
private:
    Assign(std::shared_ptr<Output> unit, std::shared_ptr<Variable> variable, std::shared_ptr<Output> value);

    size_t hashcode;
    std::shared_ptr<Output> unit;
    std::shared_ptr<Variable> variable;
    std::shared_ptr<Output> value;
    std::weak_ptr<Output> output;
};


#endif //TFL_ASSIGN_H
