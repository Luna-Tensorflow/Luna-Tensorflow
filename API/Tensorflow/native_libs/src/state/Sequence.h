//
// Created by Radek on 27.02.2019.
//

#ifndef TFL_ASSIGN_H
#define TFL_ASSIGN_H

#include "../ops/Binder.h"
#include "Variable.h"

class Sequence : public Binder {
public:
	static std::shared_ptr<Output> make_sequence(std::shared_ptr<Output> sideeffect,
																std::shared_ptr<Output> value);

	 void add_to_graph(GraphSession &graph) override;
    size_t hashcode() const override;
    std::string hash_log() const override;

private:
    Sequence(std::shared_ptr<Output> sideefect, std::shared_ptr<Output> value);

    std::shared_ptr<Output> side_effect;
    std::shared_ptr<Output> value;
    std::weak_ptr<Output> output;
    size_t hash;
};


#endif //TFL_ASSIGN_H
