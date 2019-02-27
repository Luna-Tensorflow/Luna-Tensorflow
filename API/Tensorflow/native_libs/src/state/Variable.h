//
// Created by Radek on 27.02.2019.
//

#ifndef TFL_VARIABLE_H
#define TFL_VARIABLE_H


#include "../ops/Binder.h"

class Variable : public Binder {
public:
    // TODO do we want the variable to know it's type in low-level API?
    // most likely it can be erased at this point, as we will try typechecking in Luna
    // and runtime errors shall be handled by TF anyway


private:
    Variable(std::shared_ptr<Tensor> defaultValue, std::string name);

    std::weak_ptr<Output> my_output;
};


#endif //TFL_VARIABLE_H
