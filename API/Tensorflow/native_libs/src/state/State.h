#ifndef TFL_STATE_H
#define TFL_STATE_H


#include <string>
#include <memory>
#include "../tensor/Tensor.h"

class State {
    // TODO
public:
    // TODO not sure if return by val or ptr or ?
    State updated(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> new_values) {
        return not_implemented<State>();
    }
};


#endif //TFL_STATE_H
