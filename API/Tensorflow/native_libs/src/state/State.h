#ifndef TFL_STATE_H
#define TFL_STATE_H


#include <string>
#include <memory>
#include <map>
#include "../tensor/Tensor.h"

class State {
    std::map<std::string, std::shared_ptr<Tensor>> valuation;
public:
    // TODO not sure if return by val or ptr or ?
    std::shared_ptr<State> updated(const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> &new_values) {
        // TODO
        auto s = std::make_shared<State>(*this);
        for (auto & p : new_values) {
            s->valuation[p.first] = p.second;
        }
        return s;
    }

    std::shared_ptr<State> updated(std::string name, std::shared_ptr<Tensor> value) {
        return updated({{name, value}});
    }

    static std::shared_ptr<State> make_empty() {
        return std::make_shared<State>();
    }
};


#endif //TFL_STATE_H
